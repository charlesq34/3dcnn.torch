require 'nn'
require 'cunn'
require 'cudnn'

local net = nn.Sequential()

-- conv + batchnorm + relu
local function Block(...)
    local arg = {...}
    net:add(cudnn.VolumetricConvolution(...))
    net:add(cudnn.VolumetricBatchNormalization(arg[2]))
    net:add(cudnn.ReLU(true))
    return net
end

Block(1,48,6,6,6,2,2,2)
Block(48,48,1,1,1)
Block(48,48,1,1,1)
net:add(nn.Dropout(0.2))

Block(48,160,5,5,5,2,2,2)
Block(160,160,1,1,1)
Block(160,160,1,1,1)
net:add(nn.Dropout(0.2))

Block(160,512,3,3,3,2,2,2)
Block(512,512,1,1,1)
Block(512,512,1,1,1)
net:add(nn.Dropout(0.2))

net:add(nn.View(512,8))
local t = nn.ConcatTable()
for i = 1,8 do
    local s = nn.Sequential()
    s:add(nn.Select(3,i)) -- select i-th column
    s:add(nn.Linear(512,40))
    t:add(s)
end

local w = nn.Sequential()
w:add(nn.View(4096))
w:add(nn.Linear(4096,2048))
w:add(nn.ReLU(true))
w:add(nn.Dropout(0.5))
w:add(nn.Linear(2048,2048))
w:add(nn.ReLU(true))
w:add(nn.Dropout(0.5))
w:add(nn.Linear(2048,40))
t:add(w)

net:add(t)
local function MSRinit(net)
    local function init(name)
        for k,v in pairs(net:findModules(name)) do
            local n = v.kT*v.kW*v.kH*v.nOutputPlane
            v.weight:normal(0,math.sqrt(2/n))
            v.bias:zero()
        end
    end
    init'VolumetricConvolution'
    return net
end

MSRinit(net)

criterion = nn.ParallelCriterion(true)
for i = 1,9 do
    criterion:add(nn.CrossEntropyCriterion())
end
criterion = criterion:cuda()


return net, criterion
