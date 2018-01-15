torch.manualSeed(1)
require 'cunn'
require 'optim'
matio=require 'matio'
local data = dofile('../data/synthetic/shapenetColorDRC.lua')
local gUtils = dofile('../rayUtils/grid.lua')
local netBlocks = dofile('../nnutils/netBlocks.lua')
local rp = dofile('../rayUtils/rpcolorWrapper.lua')
local netInit = dofile('../nnutils/netInit.lua')
local vUtils = dofile('../utils/visUtils.lua')
local splitUtil = dofile('../benchmark/synthetic/splits.lua')
-----------------------------
--------parameters-----------
local gridBound = 0.5 --TODO parameter fixed according to shapenet models' size
local bgDepth = 10.0 --parameter fixed according to rendering used
local bgColor = torch.Tensor({1,1,1})
unpack=unpack or table.unpack
local params = {}
--params.bgVal = 0
params.name = 'car_color'
params.gpu = 1
params.useNoise = 0
params.batchSize = 32
params.nImgs = 48 --TODO 48
params.imgSizeY = 64
params.imgSizeX = 64
params.bgWt = 0.2 -- figured out via cross-validation on the val set
params.synset = 3001627 --chair:3001627, aero:2691156, car:2958343
params.imagesave = 100
params.matsave = 1

params.gridSizeX = 32
params.gridSizeY = 32
params.gridSizeZ = 32

params.imsave = 1
params.disp = 0
params.maskOnly = 0
params.nRaysTot = 3000 --TODO 3000
params.bottleneckSize = 400
params.visIter = 1000
params.nConvEncLayers = 5
params.nConvDecLayers = 4
params.nConvEncChannelsInit = 8
params.numTrainIter = 50000
params.minDisp = 0.05
params.ip = '131.159.40.120'
params.port = 8000

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(params) do params[k] = tonumber(os.getenv(k)) or os.getenv(k) or params[k] end

params.nCams = params.nImgs
params.nRaysPerCam = torch.round(params.nRaysTot/params.nCams)

if params.disp == 0 then params.display = false else params.display = true end
if params.useNoise == 0 then params.useNoise = false end
if params.maskOnly == 0 then params.maskOnly = false end
if params.imsave == 0 then params.imsave = false end
--params.visDir = '../cachedir/visualization/' .. params.name
params.visDir = '../cachedir/visualization/' .. params.name
params.snapshotDir = '../cachedir/snapshots/shapenet/' .. params.name
params.voxelSaveDir= params.visDir .. '/vox'
params.imgSize = torch.Tensor({params.imgSizeX, params.imgSizeY})
params.gridSize = torch.Tensor({params.gridSizeX, params.gridSizeY, params.gridSizeZ})
params.synset = '0' .. tostring(params.synset) --to resolve string/number issues in passing bash arguments
--params.modelsDataDir = '../cachedir/blenderRenderPreprocess/' .. params.synset .. '/'
--params.modelsDataDir = '../../' .. params.synset .. '/'  --TODO
params.modelsDataDir='/mnt/raid/viveka/data/'..params.synset .. '/'
assert(params.minDisp < 1/bgDepth) -- otherwise we won't sample layers that don't intersect CAD model and therefore have a bad estimate of free space
print(params)
-----------------------------
-----------------------------
paths.mkdir(params.visDir)
paths.mkdir(params.snapshotDir)
cutorch.setDevice(params.gpu)
local fout = io.open(paths.concat(params.snapshotDir,'logCco2.txt'), 'w')  --TODO
for k,v in pairs(params) do
    fout:write(string.format('%s : %s\n',tostring(k),tostring(v)))
end
fout:flush()
-----------------------------
----------LossComp-----------
local minBounds = torch.Tensor({-1,-1,-1})*gridBound
local maxBounds = torch.Tensor({1,1,1})*gridBound
local step = torch.Tensor({2/params.gridSizeX,2/params.gridSizeY,2/params.gridSizeZ})*gridBound
local grid = gUtils.gridNd(minBounds, maxBounds, step)
local lossFunc = rp.rayPotential(grid, bgColor)
-----------------------------
----------Encoder------------
local encoder, nOutChannels = netBlocks.convEncoderSimple2d(params.nConvEncLayers,params.nConvEncChannelsInit,3,true) --output is nConvEncChannelsInit*pow(2,nConvEncLayers-1) X imgSize/pow(2,nConvEncLayers)
local featSpSize = params.imgSize/torch.pow(2,params.nConvEncLayers)
--print(featSpSize)
local bottleneck = nn.Sequential():add(nn.Reshape(nOutChannels*featSpSize[1]*featSpSize[2],1,1,true))
local nInputCh = nOutChannels*featSpSize[1]*featSpSize[2]
for nLayers=1,2 do --fc for joint reasoning
    bottleneck:add(nn.SpatialConvolution(nInputCh,params.bottleneckSize,1,1)):add(nn.SpatialBatchNormalization(params.bottleneckSize)):add(nn.LeakyReLU(0.2, true))
    nInputCh = params.bottleneckSize
end
encoder:add(bottleneck)
encoder:apply(netInit.weightsInit)
--print(encoder)
---------------------------------
----------World Decoder----------
local featSpSize = params.gridSize/torch.pow(2,params.nConvDecLayers)
local decoder  = nn.Sequential():add(nn.SpatialConvolution(params.bottleneckSize,nOutChannels*featSpSize[1]*featSpSize[2]*featSpSize[3],1,1,1)):add(nn.SpatialBatchNormalization(nOutChannels*featSpSize[1]*featSpSize[2]*featSpSize[3])):add(nn.ReLU(true)):add(nn.Reshape(nOutChannels,featSpSize[1],featSpSize[2],featSpSize[3],true))
decoder:add(netBlocks.convDecoderSimple3d(params.nConvDecLayers,nOutChannels,params.nConvEncChannelsInit,4,true)) --4 channels for RGB-Occupancy
decoder:add(nn.ConcatTable():add(nn.Narrow(2,1,1)):add(nn.Narrow(2,2,3)))
decoder:apply(netInit.weightsInit)
-----------------------------
----------Recons-------------
local splitUtil = dofile('../benchmark/synthetic/splits.lua')
local trainModels = splitUtil.getSplit(params.synset)['train']
local dataLoader = data.dataLoader(params.modelsDataDir, params.batchSize, params.nCams, params.nRaysPerCam, params.imgSize, params.minDisp, params.maskOnly, params.nImgs, trainModels)
local netRecons = nn.Sequential():add(encoder):add(decoder)
--local netRecons = torch.load(params.snapshotDir .. '/iter_Ccol50000.t7')
netRecons = netRecons:cuda()
--print(encoder)
--print(decoder)
local err = 0

-- Optimization parameters
local optimState = {
   learningRate = 0.0001,
   beta1 = 0.9,
}

local netParameters, netGradParameters = netRecons:getParameters()
local imgs, pred, rays

--local loss_tm = torch.Timer()
--local data_tm = torch.Timer()
--local tot_tm = torch.Timer()

-- fX required for training
local fx = function(x)
    netGradParameters:zero()
    --data_tm:reset(); data_tm:resume()
    imgs, rays = unpack(dataLoader:forward())
    
    imgs = imgs:cuda()
    pred = netRecons:forward(imgs)
    --pred = {pred[1]:clone():double(), pred[2]:clone():double()}
    --loss_tm:reset(); loss_tm:resume()
    err = lossFunc:forward(pred, rays):mean()
    --loss_tm:stop()
    local gradPred = lossFunc:backward(pred, rays)
    netRecons:backward(imgs, {gradPred[1]:cuda(), gradPred[2]:cuda()})
    return err, netGradParameters
end
--print(netRecons)
-----------------------------
----------Training-----------
if(params.display) then 
    disp = require 'display' 
    disp.configure({hostname=params.ip, port=params.port})
end
for iter=1,params.numTrainIter do
--for iter=1,1 do
    print(iter,err)
    --tot_tm:reset()
    --print(('Data/Total time : %f/%f'):format(data_tm:time().real,tm:time().real))
    fout:write(string.format('%d %f\n',iter,err))
    fout:flush()
    if(iter%params.visIter==0) then
        --print(pred)
        local dispVar2 = (1 - pred[2]):clone()
        local dispVar1 = (1 - pred[1]):clone()
        local dispVar3 = pred[2]:clone()
        --local maskDisp = dispVar2:clone()
        --local maskDisp = torch.cmul(dispVar3, torch.gt(dispVar3, 0.9):typeAs(dispVar3))
        --dispVar3 = dispVar3 - maskDisp
        if(params.disp == 1) then
            disp.image(imgs, {win=106, title='inputImC2'})
            disp.image(dispVar1:max(3):squeeze(), {win=7, title='predX1_Cocc2'})
            disp.image(dispVar1:max(4):squeeze(), {win=8, title='predY1_Cocc2'})
            disp.image(dispVar1:max(5):squeeze(), {win=9, title='predZ1_Cocc2'})
            disp.image(dispVar2:max(3):squeeze(), {win=17, title='predX2_Ccol2'})
            disp.image(dispVar2:max(4):squeeze(), {win=18, title='predY2_Ccol2'})
            disp.image(dispVar2:max(5):squeeze(), {win=19, title='predZ2_Ccol2'})
            disp.image(dispVar3:min(3):squeeze(), {win=27, title='predX3_Ccol2'})
            disp.image(dispVar3:min(4):squeeze(), {win=28, title='predY3_Ccol2'})
            disp.image(dispVar3:min(5):squeeze(), {win=29, title='predZ3_Ccol2'})
        end
        if(params.imsave == 1 and iter%params.imagesave == 0) then
            vUtils.imsave(imgs, params.visDir .. '/inputIm_Cocc'.. iter .. '.png')
            vUtils.imsave(dispVar2:max(3):squeeze(), params.visDir.. '/predX_Ccol' .. iter .. '.png')
            vUtils.imsave(dispVar2:max(4):squeeze(), params.visDir.. '/predY_Ccol' .. iter .. '.png')
            vUtils.imsave(dispVar2:max(5):squeeze(), params.visDir.. '/predZ_Ccol' .. iter .. '.png')
        end
        if(params.matsave == 1) then
            local vox_dir=params.voxelSaveDir.. tostring(iter)
            paths.mkdir(vox_dir)
            for i =1,params.batchSize do
                matio.save(vox_dir ..  string.format('/pred_%03d.mat',i),pred[2][i]:float())
            end
            print ('All voxels saved')
        end
    end
    if(iter%5000)==0 then
        torch.save(params.snapshotDir .. '/iter_Ccol'.. iter .. '.t7', netRecons)
    end
    optim.adam(fx, netParameters, optimState)
    --print(tot_tm:time().real, data_tm:time().real, loss_tm:time().real) 
end
