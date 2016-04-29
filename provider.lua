require 'torch'
require 'hdf5'

-- download dataset 60x azimuth+elevation augmented
if not paths.dirp('data/modelnet40_60x') then
    local www = 'https://shapenet.cs.stanford.edu/media/modelnet40_h5.tar'
    local tar = paths.basename(www)
    os.execute('cd data')
    os.execute('wget ' .. www .. '; ' .. 'tar xvf ' .. tar)
    os.execute('cd ..')
end

-- small jitter data augmentation
-- input: 5D tensor of NxCxDxHxW
math.randomseed(123)
function jitter_chunk(src,jitter)
    dst = torch.zeros(src:size())
    for idx =1,src:size()[1] do
        local i = math.random(-jitter, jitter)
        local j = math.random(-jitter, jitter)
        local k = math.random(-jitter, jitter)
        --print(dst[{{},{},{i+1,dst:size()[3]},{},{}}]:squeeze())
        if i > 0 then dst[{{idx},{},{i+1,dst:size()[3]},{},{}}] = src[{{idx},{},{1,dst:size()[3]-i},{},{}}] end
        if i < 0 then dst[{{idx},{},{1,dst:size()[3]+i},{},{}}] = src[{{idx},{},{-i+1,dst:size()[3]},{},{}}] end
        if j > 0 then dst[{{idx},{},{},{j+1,dst:size()[4]},{}}] = src[{{idx},{},{},{1,dst:size()[4]-j},{}}] end
        if j < 0 then dst[{{idx},{},{},{1,dst:size()[4]+j},{}}] = src[{{idx},{},{},{-j+1,dst:size()[4]},{}}] end
        if k > 0 then dst[{{idx},{},{},{},{k+1,dst:size()[5]}}] = src[{{idx},{},{},{},{1,dst:size()[5]-k}}] end
        if k < 0 then dst[{{idx},{},{},{},{1,dst:size()[5]+k}}] = src[{{idx},{},{},{},{-k+1,dst:size()[5]}}] end
    end
    return dst
end

-- read h5 filename list
function getDataFiles(input_file)
    local train_files = {}
    for line in io.lines(input_file) do
        train_files[#train_files+1] = line
    end
    return train_files
end

-- load h5 file data into memory
function loadDataFile(file_name)
    local current_file = hdf5.open(file_name,'r')
    local current_data = current_file:read('data'):all():float()
    local current_label = torch.squeeze(current_file:read('label'):all():add(1))
    current_file:close()
    return current_data, current_label
end



