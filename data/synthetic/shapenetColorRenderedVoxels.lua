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
    self.gen=torch.Generator()
    torch.manualSeed(self.gen, 0)
    return self
end

function dataLoader:forward()
    local fileNames = {'e30.000000_a0.000000', 'e30.000000_a15.000000', 'e30.000000_a30.000000', 'e30.000000_a45.000000', 'e30.000000_a60.000000', 'e30.000000_a75.000000', 'e30.000000_a90.000000', 'e30.000000_a105.000000', 'e30.000000_a120.000000', 'e30.000000_a135.000000', 'e30.000000_a150.000000', 'e30.000000_a165.000000', 'e30.000000_a180.000000', 'e30.000000_a195.000000', 'e30.000000_a210.000000', 'e30.000000_a225.000000', 'e30.000000_a240.000000', 'e30.000000_a255.000000', 'e30.000000_a270.000000', 'e30.000000_a285.000000', 'e30.000000_a300.000000', 'e30.000000_a315.000000', 'e30.000000_a330.000000', 'e30.000000_a345.000000', 'e-30.000000_a360.000000', 'e-30.000000_a375.000000', 'e-30.000000_a390.000000', 'e-30.000000_a405.000000', 'e-30.000000_a420.000000', 'e-30.000000_a435.000000', 'e-30.000000_a450.000000', 'e-30.000000_a465.000000', 'e-30.000000_a480.000000', 'e-30.000000_a495.000000', 'e-30.000000_a510.000000', 'e-30.000000_a525.000000', 'e-30.000000_a540.000000', 'e-30.000000_a555.000000', 'e-30.000000_a570.000000', 'e-30.000000_a585.000000', 'e-30.000000_a600.000000', 'e-30.000000_a615.000000', 'e-30.000000_a630.000000', 'e-30.000000_a645.000000', 'e-30.000000_a660.000000', 'e-30.000000_a675.000000', 'e-30.000000_a690.000000', 'e-30.000000_a705.000000'}
 
    local imgs = torch.Tensor(self.bs, 3, self.imgSize[1], self.imgSize[2]):fill(0)
    local voxels = torch.Tensor(self.bs, 3, self.voxelSize[1], self.voxelSize[2], self.voxelSize[3]):fill(0)
    for b = 1,self.bs do
        local mId = torch.random( self.gen, 1,#self.modelNames)
        local imgsDir = paths.concat(self.imgsDir, self.modelNames[mId]) 
        --local nImgs = #BuildArray(paths.files(imgsDir,'.mat'))
        --local inpImgNum = torch.random(0,nImgs-1)
        local inpImgNum = torch.random(self.gen,1,#fileNames)
        local imgRgb = image.load(string.format('%s/image_%s.png',imgsDir,fileNames[inpImgNum]))  --TODO try masks instead of imgs
        --local imgRgb = image.load(string.format('%s/render_%d.png',imgsDir,inpImgNum))
        if(self.bGImgsList) then
            -- useful for PASCAL VOC experiments, we'll set the bgImgsList externally
            imgRgb = cropUtils.blendBg(imgRgb, self.bgImgsList)
            imgRgb = image.scale(imgRgb,self.imgSize[2], self.imgSize[1])
        else
            imgRgb = image.scale(imgRgb,self.imgSize[2], self.imgSize[1])
            --print(imgRgb:size())
            --local alphaMask = imgRgb[3]:repeatTensor(3,1,1) --when imgRgb[3] is used then it works but it may be conceptually wrong
            --imgRgb = torch.cmul(imgRgb:narrow(1,1,3),alphaMask) + 1 - alphaMask
        end
        imgs[b] = imgRgb
        
        local voxelFile = paths.concat(self.voxelsDir, self.modelNames[mId] .. '/model_32.mat')
        --local voxelFile = paths.concat(self.voxelsDir, self.modelNames[mId] .. '.mat')
        voxels[b] = matio.load(voxelFile,{'voxel'})['voxel']:typeAs(voxels)
        --voxels[b][1] = matio.load(voxelFile,{'Volume'})['Volume']:typeAs(voxels)
    end
    --voxels = voxels:transpose(5,4) --to match things done when rendering via blender --TODO do we need it? 
    -- blender use right hand coord.
    -- technically, we also need to flip values in Z but symmetry should take care of it
    return imgs, voxels
end
-------------------------------
-------------------------------
M.dataLoader = dataLoader
return M
