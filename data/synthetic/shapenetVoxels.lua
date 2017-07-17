local M = {}
require 'image'
local cropUtils = dofile('../utils/cropUtils.lua')
local matio = require 'matio'
--local pcl = require 'pcl'
--local pclUtils = dofile('../utils/pclUtils.lua')
-------------------------------
-------------------------------
local function BuildArray(...)
  local arr = {}
  for v in ... do
    arr[#arr + 1] = v
  end
  return arr
end
-------------------------------
-------------------------------
local dataLoader = {}
dataLoader.__index = dataLoader

setmetatable(dataLoader, {
    __call = function (cls, ...)
        return cls.new(...)
    end,
})

function dataLoader.new(imgsDir, voxelsDir, bs, imgSize, voxelSize, modelNames)
    local self = setmetatable({}, dataLoader)
    self.bs = bs
    self.imgSize = imgSize
    self.imgsDir = imgsDir
    self.voxelSize = voxelSize
    self.voxelsDir = voxelsDir
    self.modelNames = modelNames
    return self
end

function dataLoader:forward()
    local imgs = torch.Tensor(self.bs, 3, self.imgSize[1], self.imgSize[2]):fill(0)
    local voxels = torch.Tensor(self.bs, 1, self.voxelSize[1], self.voxelSize[2], self.voxelSize[3]):fill(0)
    for b = 1,self.bs do
        local mId = torch.random(1,#self.modelNames)
        local imgsDir = paths.concat(self.imgsDir, self.modelNames[mId]) 
        --local nImgs = #BuildArray(paths.files(imgsDir,'.mat'))
        --local inpImgNum = torch.random(0,nImgs-1)
        local imgAzimuthal = {'015', '030', '045', '060', '075', '090', '105', '120', '135', '150', '165', '180', '195', '210', '225', '240', '255', '270', '285', '300', '315', '330', '345', '360'}
        local inpImgNum = torch.random(1,#imgAzimuthal)
        local imgRgb = image.load(string.format('%s/imgs/a%s_e030.jpg',imgsDir,imgAzimuthal[inpImgNum]))  --TODO try masks instead of imgs
        --local imgRgb = image.load(string.format('%s/render_%d.png',imgsDir,inpImgNum))
        if(self.bGImgsList) then
            -- useful for PASCAL VOC experiments, we'll set the bgImgsList externally
            imgRgb = cropUtils.blendBg(imgRgb, self.bgImgsList)
            imgRgb = image.scale(imgRgb,self.imgSize[2], self.imgSize[1])
        else
            imgRgb = image.scale(imgRgb,self.imgSize[2], self.imgSize[1])
            --print(imgRgb:size())
            local alphaMask = imgRgb[3]:repeatTensor(3,1,1) --when imgRgb[3] is used then it works but it may be conceptually wrong
            imgRgb = torch.cmul(imgRgb:narrow(1,1,3),alphaMask) + 1 - alphaMask
        end
        imgs[b] = imgRgb
        
        local voxelFile = paths.concat(self.voxelsDir, self.modelNames[mId] .. '/model_32.mat')
        --local voxelFile = paths.concat(self.voxelsDir, self.modelNames[mId] .. '.mat')
        voxels[b][1] = matio.load(voxelFile,{'voxel'})['voxel']:typeAs(voxels)
        --voxels[b][1] = matio.load(voxelFile,{'Volume'})['Volume']:typeAs(voxels)
    end
    voxels = voxels:transpose(5,4) --to match things done when rendering via blender --TODO do we need it? 
    -- blender use right hand coord.
    -- technically, we also need to flip values in Z but symmetry should take care of it
    return imgs, voxels
end
-------------------------------
-------------------------------
M.dataLoader = dataLoader
return M
