require 'nn'
require 'stn3d'
require 'cunn'
require 'cudnn'
require 'optim'
require 'xlua'
dofile './provider.lua'


opt = lapp[[
    -s,--save               (default "mo_logs")                     subdirectory to save logs
    -b,--batchSize          (default 64)                            batch size
    -r,--learningRate       (default 0.001)                         learning rate
    --learningRateDecay     (default 1e-7)                          learning rate decay
    --weigthDecay           (default 0.0005)                        weight decay
    -m,--momentum           (default 0.9)                           mementum
    --epoch_step            (default 20)                            epoch step
    -g,--gpu_index          (default 1)                             GPU index
    --max_epoch             (default 50)                            maximum number of epochs
    --model                 (default 3dnin_fc)                      model name
    --model_param_file      (default "torch_models/3dnin_fc.net)    model parameter filename
    --pool_layer_idx        (default -1)                            pool output of the idx-th layer
    --train_data            (default "data/modelnet40_20x_stack/train_data.txt")   txt file containing train h5 filenames
    --test_data             (default "data/modelnet40_20x_stack/test_data.txt")    txt file containing test h5 filenames
]]

print(opt)

-- set gpu
cutorch.setDevice(opt.gpu_index)


print('Loading model...')
model = torch.load(opt.model_param_file):cuda()
print(model)

if opt.pool_layer_idx < 1 then
    print('Select max pooling from which layer\'s output, type in layer index:')
    layer_idx = tonumber(io.read())
    print(layer_idx)
else
    layer_idx = opt.pool_layer_idx
end

print('Loading data...')
train_files = getDataFiles(opt.train_data)
test_files = getDataFiles(opt.test_data)
print(train_files)
print(test_files)

-- Extract train set features
train_data = {}
train_label = {}
train_cnt = 1
for file_idx = 1, #train_files do
    current_data, current_label = loadDataFile(train_files[file_idx])
    for t = 1,current_data:size(1) do
        xlua.progress(t, current_data:size(1))
        local inputs = current_data[t]:reshape(20,1,30,30,30) -- stack size is 20
        target = current_label[t]
        model:forward(inputs:cuda())
        local features = model:get(layer_idx).output
        max_pooled_feature = torch.max(features,1)
        
        train_data[train_cnt] = max_pooled_feature
        train_label[train_cnt] = target
        train_cnt = train_cnt + 1
    end
end

-- Extract test set features
test_data = {}
test_label = {}
test_cnt = 1
for file_idx = 1, #test_files do
    current_data, current_label = loadDataFile(test_files[file_idx])
    for t = 1,current_data:size(1) do
        xlua.progress(t, current_data:size(1))
        local inputs = current_data[t]:reshape(20,1,30,30,30)
        target = current_label[t]
        model:forward(inputs:cuda())
        local features = model:get(layer_idx).output
        max_pooled_feature = torch.max(features,1)
        
        test_data[test_cnt] = max_pooled_feature
        test_label[test_cnt] = target
        test_cnt = test_cnt + 1
    end
end
print(#train_data)
print(#test_data)

print('Starting to train multi-orientation pooling ...')

-- pooling model
model_new = model:clone()
for i = 1,layer_idx do
    model_new:remove(1)
end
model = model_new:cuda()
model:zeroGradParameters()
parameters, gradParameters = model:getParameters()
print(model)

unused, criterion = dofile('torch_models/'..opt.model..'.lua')
if not criterion then
    criterion = nn.CrossEntropyCriterion():cuda()
end

-- config for SGD solver
optimState = {
    learningRate = opt.learningRate,
    weightDecay = 0.00005,
    momentum = 0.9,
    learningRateDecay = 1e-7,
}
-- config logging
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
testLogger:setNames{'% mean class accuracy (train set)', '% mean class accuracy (test set)'}
testLogger.showPlot = 'false'

-- confusion matrix
confusion = optim.ConfusionMatrix(40)
confusion:zero()

epoch_step = opt.epoch_step
batchSize = opt.batchSize
function train()
    model:training()
    epoch = epoch or 1 -- if epoch not defined, assign it as 1
    
    if epoch % epoch_step == 0 then optimState.learningRate = optimState.learningRate/2 end

    local tic = torch.tic()
    local filesize = #train_data
    local targets = torch.CudaTensor(batchSize)
    local indices = torch.randperm(filesize):long():split(batchSize)
    -- remove last mini-batch so that all the batches have equal size
    indices[#indices] = nil 

    for t, v in ipairs(indices) do 
        xlua.progress(t, #indices)

        local inputs = train_data[v[1]]
        for i = 2,batchSize do
            inputs = torch.cat(inputs, train_data[v[i]],1)
        end
        for i = 1,batchSize do
            targets[i] = train_label[v[i]]
        end

        -- a function that takes single input and return f(x) and df/dx
        local feval = function(x)
            if x ~= parameters then parameters:copy(x) end
            gradParameters:zero()

            local outputs = model:forward(inputs)
            local f = criterion:forward(outputs, targets)
            local df_do = criterion:backward(outputs, targets)
            model:backward(inputs, df_do) -- gradParameters in model have been updated

            if torch.type(outputs) == 'table' then -- multiple outputs, take the last one
                confusion:batchAdd(outputs[#outputs], targets)
            else
                confusion:batchAdd(outputs, targets)    
            end
            return f, gradParameters
        end

        -- use SGD optimizer: parameters as input to feval will be updated
        optim.sgd(feval, parameters, optimState)
    end
    
    confusion:updateValids()
    print(('Train accuracy: '..'%.2f'..' %%\t time: %.2f s'):format(
            confusion.totalValid * 100, torch.toc(tic)))

    train_acc = confusion.totalValid * 100

    confusion:zero()
    epoch = epoch + 1
end

function test()
    -- disable flips, dropouts and batch normalization
    model:evaluate()
    
    local filesize = #test_data
    local indices = torch.randperm(filesize):long():split(batchSize)

    for t, v in ipairs(indices) do
        local inputs = test_data[v[1]]
        for i = 2,v:size(1) do
            inputs = torch.cat(inputs, test_data[v[i]],1)
        end
        local targets = torch.CudaTensor(v:size(1))
        for i = 1,v:size(1) do
            targets[i] = test_label[v[i]]
        end
        
        local outputs = model:forward(inputs)

        if torch.type(outputs) == 'table' then -- multiple outputs, take the last one
            confusion:batchAdd(outputs[#outputs], targets)
        else
            confusion:batchAdd(outputs, targets)    
        end
    end
    confusion:updateValids()
    print('Test accuracy:', confusion.totalValid * 100)

    -- logging test result to txt and html files
    if testLogger then
        paths.mkdir(opt.save)
        testLogger:add{train_acc, confusion.totalValid * 100}
        testLogger:style{'-','-'}
        testLogger:plot()

        local base64im
        do
          os.execute(('convert -density 200 %s/test.log.eps %s/test.png'):format(opt.save,opt.save))
          os.execute(('openssl base64 -in %s/test.png -out %s/test.base64'):format(opt.save,opt.save))
          local f = io.open(opt.save..'/test.base64')
          if f then base64im = f:read'*all' end
        end

        local file = io.open(opt.save..'/report.html','w')
        file:write(([[
        <!DOCTYPE html>
        <html>
        <body>
        <title>%s - %s</title>
        <img src="data:image/png;base64,%s">
        <h4>optimState:</h4>
        <table>
        ]]):format(opt.save,epoch,base64im))
        for k,v in pairs(optimState) do
          if torch.type(v) == 'number' then
            file:write('<tr><td>'..k..'</td><td>'..v..'</td></tr>\n')
          end
        end
        file:write'</table><pre>\n'
        file:write(tostring(confusion)..'\n')
        file:write(tostring(model)..'\n')
        file:write'</pre></body></html>'
        file:close()
    end

    -- save model every 10 epochs
    if epoch % 10 == 0 then
      local filename = paths.concat(opt.save, 'model.net')
      print('==> saving model to '..filename)
      -- torch.save(filename, model:get(3):clearState())
      torch.save(filename, model)
    end
    -- save model every 10 epochs
    if epoch % 10 == 0 then
      local filename = paths.concat(opt.save, 'model.net')
      print('==> saving model to '..filename)
      -- torch.save(filename, model:get(3):clearState())
      torch.save(filename, model)
    end 
    
    confusion:zero()
end

for e = 1,opt.max_epoch do
    train()
    test()
end
