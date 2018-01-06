torch.manualSeed(1)
require 'cunn'
require 'optim'
matio=require 'matio'
local data = dofile('../data/synthetic/shapenetColorVoxels.lua')
--local data = dofile('../data/synthetic/shapenetColorRenderedVoxels.lua')
local netBlocks = dofile('../nnutils/netBlocks.lua')
local netInit = dofile('../nnutils/netInit.lua')
local vUtils = dofile('../utils/visUtils.lua')
-----------------------------
--------parameters-----------
local params = {}
--params.bgVal = 0
params.name = 'shapenetVoxels'
params.gpu = 1
params.batchSize = 32
params.imgSizeY = 64
params.imgSizeX = 64
params.synset = 2958343 --chair:3001627, aero:2691156, car:2958343

params.gridSizeX = 32
params.gridSizeY = 32
params.gridSizeZ = 32
params.lambda=0.9
params.matsave=1
params.imsave = 0
params.disp = 0
params.bottleneckSize = 400
params.visIter = 1000
params.nConvEncLayers = 5
params.nConvDecLayers = 4
params.nConvEncChannelsInit = 8
params.nVoxelChannels = 3
params.nOccChannels = 1
params.numTrainIter = 500000
params.ip = '131.159.40.120'
params.port = 8000
-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(params) do params[k] = tonumber(os.getenv(k)) or os.getenv(k) or params[k] end
if params.disp == 0 then params.display = false else params.display = true end
if params.imsave == 0 then params.imsave = false end
params.name = params.name .. tostring(params.lambda)
params.visDir = '../cachedir/visualization/' .. params.name
params.snapshotDir = '../cachedir/snapshots/shapenet/' .. params.name
params.imgSize = torch.Tensor({params.imgSizeX, params.imgSizeY})
params.gridSize = torch.Tensor({params.gridSizeX, params.gridSizeY, params.gridSizeZ})
params.synset = '0' .. tostring(params.synset) --to resolve string/number issues in passing bash arguments
--params.modelsDataDir = '../cachedir/blenderRenderPreprocess/' .. params.synset .. '/'
params.modelsDataDir = '../../../arnab/nips16_PTN/data/shapenetcore_viewdata/' .. params.synset .. '/'
--params.modelsDataDir='/mnt/raid/viveka/data/'..params.synset .. '/'
--params.voxelsDir = '../cachedir/shapenet/modelVoxels/' .. params.synset .. '/'
params.voxelsDir = '../../../arnab/nips16_PTN/data/shapenetcore_colvoxdata/' .. params.synset .. '/'
params.voxelSaveDir= params.visDir .. '/vox'
print(params)
-----------------------------
-----------------------------
paths.mkdir(params.visDir)
paths.mkdir(params.snapshotDir)
cutorch.setDevice(params.gpu)
local fout = io.open(paths.concat(params.snapshotDir,'log.txt'), 'w')
for k,v in pairs(params) do
    fout:write(string.format('%s : %s\n',tostring(k),tostring(v)))
end
fout:flush()
-----------------------------
----------LossComp-----------
local lossFunc = nn.BCECriterion()
local colLossFunc = nn.MSECriterion()
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
decoder:add(netBlocks.convDecoderSimple3dHeads(params.nConvDecLayers,nOutChannels,params.nConvEncChannelsInit,params.nVoxelChannels,params.nOccChannels,true))
decoder:apply(netInit.weightsInit)
-----------------------------
----------Recons-------------
local splitUtil = dofile('../benchmark/synthetic/splits.lua')
local trainModels = splitUtil.getSplit(params.synset)['train']
local testModels = splitUtil.getSplit(params.synset)['test']
--local trainModels = {trainModels[1]}
--print(trainModels)
local dataLoader = data.dataLoader(params.modelsDataDir, params.voxelsDir, params.batchSize, params.imgSize, params.gridSize, trainModels)
local dataLoaderTest = data.dataLoader(params.modelsDataDir, params.voxelsDir, params.batchSize, params.imgSize, params.gridSize, testModels)
local netRecons = nn.Sequential():add(encoder):add(decoder)
--local netRecons = torch.load(params.snapshotDir .. '/iter10000.t7')
print(netRecons)
netRecons = netRecons:cuda()
lossFunc = lossFunc:cuda()
colLossFunc = colLossFunc:cuda()
--print(encoder)
--print(decoder)
local err = 0

-- Optimization parameters
local optimState = {
   learningRate = 0.0001,
   beta1 = 0.9,
}

local netParameters, netGradParameters = netRecons:getParameters()
local tm = torch.Timer()
local data_tm = torch.Timer()
local imgs, pred, rays

-- fX required for training
local fx = function(x)
    tm:reset(); tm:resume()
    netGradParameters:zero()
    data_tm:reset(); data_tm:resume()
    imgs, voxelsGt = dataLoader:forward()
    data_tm:stop()
    --print('Data loaded')
    local voxelsOcc=torch.sum(voxelsGt,2)
    voxelsOcc:apply( function(x) 
      if x>2.99 then return 0
      else return 1
      end 
    end)

    local occMask=torch.repeatTensor(voxelsOcc,1,3,1,1,1) 

    imgs = imgs:cuda()
    voxelsGt = voxelsGt:cuda()
    voxelsOcc= voxelsOcc:cuda()
    occMask = occMask:cuda()
    netRecons:forward(imgs)
    color=netRecons.output[1]
    pred=netRecons.output[2]

    color:cmul(occMask)
    voxelsGt:cmul(occMask)
    err = (1-params.lambda)*lossFunc:forward(pred, voxelsOcc)
    err = err + (params.lambda)*colLossFunc:forward(color,voxelsGt)
    local gradPred = (1-params.lambda)*lossFunc:backward(pred, voxelsOcc)
    local gradColor = (params.lambda)*colLossFunc:backward(color,voxelsGt)
    netRecons:backward(imgs, { gradColor , gradPred})
    tm:stop()
    return err, netGradParameters
end


local err_test=0
function eval()
    netRecons:evaluate()
    imgs, voxelsGt = dataLoader:forward()
    local voxelsOcc=torch.sum(voxelsGt,2)
    voxelsOcc:apply( function(x) 
      if x>2.99 then return 0
      else return 1
      end 
    end)

    local occMask=torch.repeatTensor(voxelsOcc,1,3,1,1,1) 
    imgs = imgs:cuda()
    voxelsGt = voxelsGt:cuda()
    voxelsOcc= voxelsOcc:cuda()
    occMask = occMask:cuda()
    netRecons:forward(imgs)
    color=netRecons.output[1]
    pred=netRecons.output[2]
    color:cmul(occMask)
    voxelsGt:cmul(occMask)
    err_test = (1-params.lambda)*lossFunc:forward(pred, voxelsOcc)
    err_test = err_test + params.lambda*colLossFunc:forward(color,voxelsGt)
    return err_test 
end

--print(netRecons)
-----------------------------
----------Training-----------
if(params.display) then 
    disp = require 'display' 
    disp.configure({hostname=params.ip, port=params.port})
end
local forwIter = 0
train_err={}
test_err={}
for iter=1,params.numTrainIter do
    print(iter,err)
    --print(('Data/Total time : %f/%f'):format(data_tm:time().real,tm:time().real))
    fout:write(string.format('%d %f\n',iter,err))
    table.insert(train_err,err)
    fout:flush()
    if(iter%params.visIter==0) then
        if(params.disp == 1) then
            disp.image(imgs, {win=1000, title='inputIm'})
            disp.image(color:max(3):squeeze(), {win=1, title='predX'})
            disp.image(color:max(4):squeeze(), {win=2, title='predY'})
            disp.image(color:max(5):squeeze(), {win=3, title='predZ'})
            
            disp.image(voxelsGt:max(3):squeeze(), {win=4, title='gtX'})
            disp.image(voxelsGt:max(4):squeeze(), {win=5, title='gtY'})
            disp.image(voxelsGt:max(5):squeeze(), {win=6, title='gtZ'})
            
            disp.image(pred:max(3):squeeze(), {win=7, title='occX'})
            disp.image(pred:max(4):squeeze(), {win=8, title='occY'})
            disp.image(pred:max(5):squeeze(), {win=9, title='occZ'})
            
            disp.plot(train_err,{win=10,title='Training Error'})
           
            table.insert(test_err,eval())

            disp.image(imgs, {win=100, title='Test-inputIm'})
            disp.image(color:max(3):squeeze(), {win=11, title='Test-predX'})
            disp.image(color:max(4):squeeze(), {win=12, title='Test-predY'})
            disp.image(color:max(5):squeeze(), {win=13, title='Test-predZ'})
            
            disp.image(voxelsGt:max(3):squeeze(), {win=14, title='Test-gtX'})
            disp.image(voxelsGt:max(4):squeeze(), {win=15, title='Test-gtY'})
            disp.image(voxelsGt:max(5):squeeze(), {win=16, title='Test-gtZ'})
            
            disp.image(pred:max(3):squeeze(), {win=17, title='Test-occX'})
            disp.image(pred:max(4):squeeze(), {win=18, title='Test-occY'})
            disp.image(pred:max(5):squeeze(), {win=19, title='Test-occZ'})
            
            disp.plot(test_err,{win=20,title='Test Error'})
        end
        if(params.imsave == 1) then
            vUtils.imsave(imgs, params.visDir .. '/inputIm'.. iter .. '.png')
            vUtils.imsave(color:max(3):squeeze(), params.visDir.. '/predX' .. iter .. '.png')
            vUtils.imsave(color:max(4):squeeze(), params.visDir.. '/predY' .. iter .. '.png')
            vUtils.imsave(color:max(5):squeeze(), params.visDir.. '/predZ' .. iter .. '.png')

            vUtils.imsave(pred:max(3):squeeze(), params.visDir.. '/occX' .. iter .. '.png')
            vUtils.imsave(pred:max(4):squeeze(), params.visDir.. '/occY' .. iter .. '.png')
            vUtils.imsave(pred:max(5):squeeze(), params.visDir.. '/occZ' .. iter .. '.png')
        end
        if(params.matsave==1) then
            local vox_dir=params.voxelSaveDir.. tostring(iter)
            paths.mkdir(vox_dir)
            for i =1,params.batchSize do
                matio.save(vox_dir .. string.format('/gt_%03d.mat',i),voxelsGt[i]:float())
                matio.save(vox_dir ..  string.format('/pred_%03d.mat',i),color[i]:float())
                matio.save(vox_dir ..  string.format('/pred_occ%03d.mat',i),pred[i]:float())
            end
        end
    end
    if(iter%5000)==0 then
        torch.save(params.snapshotDir .. '/iter'.. iter .. '.t7', netRecons)
    end
    optim.adam(fx, netParameters, optimState)
end
