require 'torch'
require 'hdf5'

-- download dataset 60x azimuth+elevation augmented
if not paths.dirp('data/modelnet40_60x') then
    local www = 'https://shapenet.cs.stanford.edu/media/modelnet40_h5.tar'
    local tar = paths.basename(www)
    os.execute('mkdir data')
    os.execute('wget ' .. www .. '; ' .. 'tar xvf ' .. tar)
    os.execute('mv modelnet40_* data')
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
        if i >= 0 then xidx = {i+1,dst:size(3),1,dst:size(3)-i} end
        if i < 0 then xidx = {1,dst:size(3)+i,-i+1,dst:size(3)} end
        if j >= 0 then yidx = {j+1,dst:size(4),1,dst:size(4)-j} end
        if j < 0 then yidx = {1,dst:size(4)+j,-j+1,dst:size(4)} end
        if k >= 0 then zidx = {k+1,dst:size(5),1,dst:size(5)-k} end
        if k < 0 then zidx = {1,dst:size(5)+k,-k+1,dst:size(5)} end
        dst[{{idx},{},{xidx[1],xidx[2]},{yidx[1],yidx[2]},{zidx[1],zidx[2]}}] = 
                src[{{idx},{},{xidx[3],xidx[4]},{yidx[3],yidx[4]},{zidx[3],zidx[4]}}]

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



