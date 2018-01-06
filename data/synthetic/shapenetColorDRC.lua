local M = {}
require 'image'
local cropUtils = dofile('../utils/cropUtils.lua')
local dUtils = dofile('../utils/depthUtils.lua')
local matio = require 'matio'
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

function dataLoader.new(synsetDir, bs, nCams, nRaysPerCam, imgSize, minDisp, maskOnly, nImgs, modelNames)
    local self = setmetatable({}, dataLoader)
    self.bs = bs
    self.nCams = nCams
    self.nRaysPerCam = nRaysPerCam
    self.imgSize = imgSize
    self.synsetDir = synsetDir
    self.nImgs = nImgs
    --print(self.synsetDir)
    self.modelNames = modelNames
    return self
end

function dataLoader:forward()
    
    local fileNames = {'e30.000000_a0.000000', 'e30.000000_a15.000000', 'e30.000000_a30.000000', 'e30.000000_a45.000000', 'e30.000000_a60.000000', 'e30.000000_a75.000000', 'e30.000000_a90.000000', 'e30.000000_a105.000000', 'e30.000000_a120.000000', 'e30.000000_a135.000000', 'e30.000000_a150.000000', 'e30.000000_a165.000000', 'e30.000000_a180.000000', 'e30.000000_a195.000000', 'e30.000000_a210.000000', 'e30.000000_a225.000000', 'e30.000000_a240.000000', 'e30.000000_a255.000000', 'e30.000000_a270.000000', 'e30.000000_a285.000000', 'e30.000000_a300.000000', 'e30.000000_a315.000000', 'e30.000000_a330.000000', 'e30.000000_a345.000000', 'e-30.000000_a360.000000', 'e-30.000000_a375.000000', 'e-30.000000_a390.000000', 'e-30.000000_a405.000000', 'e-30.000000_a420.000000', 'e-30.000000_a435.000000', 'e-30.000000_a450.000000', 'e-30.000000_a465.000000', 'e-30.000000_a480.000000', 'e-30.000000_a495.000000', 'e-30.000000_a510.000000', 'e-30.000000_a525.000000', 'e-30.000000_a540.000000', 'e-30.000000_a555.000000', 'e-30.000000_a570.000000', 'e-30.000000_a585.000000', 'e-30.000000_a600.000000', 'e-30.000000_a615.000000', 'e-30.000000_a630.000000', 'e-30.000000_a645.000000', 'e-30.000000_a660.000000', 'e-30.000000_a675.000000', 'e-30.000000_a690.000000', 'e-30.000000_a705.000000'}
    local nRays = self.nRaysPerCam*self.nCams
    local imgs = torch.Tensor(self.bs, 3, self.imgSize[1], self.imgSize[2]):fill(0)
    local colors = torch.Tensor(self.bs, nRays, 3):float()
    local origins = torch.Tensor(self.bs, nRays, 3):float()
    local directions = torch.Tensor(self.bs, nRays, 3):float()
    for b = 1,self.bs do
        local mId = torch.random(1,#self.modelNames)
        --print(self.modelNames[mId])i
        --local imgsDir = paths.concat(self.synsetDir, 'c951e7bd4c0cbce17ec5a98b3b8c425f')
        local imgsDir = paths.concat(self.synsetDir, self.modelNames[mId])
        local nImgs = self.nImgs or #BuildArray(paths.files(imgsDir,'.mat'))
        --local inpImgNum = torch.random(0,nImgs-1)
        local inpImgNum = torch.random(1,nImgs)
        --print(string.format('%s/render_%d.png',imgsDir,inpImgNum))
        --local imgRgb = image.load(string.format('%s/render_%d.png',imgsDir,inpImgNum))
        local imgRgb = image.load(string.format('%s/image_%s.png',imgsDir,fileNames[inpImgNum]))
        if(self.bgImgsList) then
            -- useful for PASCAL VOC experiments, we'll set the bgImgsList externally
            imgRgb = cropUtils.blendBg(imgRgb, self.bgImgsList)
            imgRgb = image.scale(imgRgb,self.imgSize[2], self.imgSize[1])
        else
            imgRgb = image.scale(imgRgb,self.imgSize[2], self.imgSize[1])
            --local alphaMask = imgRgb[4]:repeatTensor(3,1,1)
            --imgRgb = torch.cmul(imgRgb:narrow(1,1,3),alphaMask) + 1 - alphaMask
        end
        imgs[b] = imgRgb
        local rPerm = torch.randperm(nImgs)
        
        for nc = 1,self.nCams do
            local numSamples =  self.nRaysPerCam
            --local imgNum = rPerm[nc] - 1
            local imgNum = rPerm[nc]
            --local imgNum = (nc==1) and inpImgNum or torch.random(0,nImgs-1)
            --print(string.format('%s/camera_%d.mat',imgsDir,imgNum))
            --local camData = matio.load(string.format('%s/camera_%d.mat',imgsDir,imgNum),{'pos','quat','K','extrinsic'})
            local camData = matio.load(string.format('%s/camera_%s.mat',imgsDir,fileNames[imgNum]),{'pos','K','extrinsic'})
            --print(string.format('%s/%s_%d.png',imgsDir,'render',imgNum))
            --local colorIm = image.load(string.format('%s/%s_%d.png',imgsDir,'render',imgNum))
            --print('done')
            local colorIm = image.load(string.format('%s/image_%s.png',imgsDir,fileNames[imgNum]))
            --local alphaMask = colorIm[4]:repeatTensor(3,1,1)
            --colorIm = torch.cmul(colorIm:narrow(1,1,3),alphaMask) + 1 - alphaMask
            
            local mMat = dUtils.inverseMotionMat(camData.extrinsic)
            
            local dirSamples, colorSamples = dUtils.sampleRaysColor(colorIm, camData.K, numSamples)
            
            local orgSamples = torch.Tensor(4,numSamples):fill(0)
            orgSamples:narrow(1,4,1):fill(1)
            orgSamples = torch.mm(mMat, orgSamples:typeAs(mMat)):narrow(1,1,3):transpose(2,1)
            dirSamples = torch.mm(mMat, dirSamples:typeAs(mMat)):narrow(1,1,3):transpose(2,1)
            
            colors[b]:narrow(1,(nc-1)*numSamples+1,numSamples):copy(colorSamples)
            directions[b]:narrow(1,(nc-1)*numSamples+1,numSamples):copy(dirSamples)
            origins[b]:narrow(1,(nc-1)*numSamples+1,numSamples):copy(orgSamples)
        end
    end
    return {imgs, {origins, directions, colors}}
end
-------------------------------
-------------------------------
M.dataLoader = dataLoader
return M
