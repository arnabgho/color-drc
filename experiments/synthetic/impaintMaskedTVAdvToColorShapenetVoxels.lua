torch.manualSeed(1)
require 'nn'
require 'cunn'
require 'optim'
matio=require 'matio'
--local data = dofile('../data/synthetic/shapenetColorVoxels.lua')
local data = dofile('../data/synthetic/shapenetColorRenderedVoxels.lua')
local netBlocks = dofile('../nnutils/netBlocks.lua')
local netInit = dofile('../nnutils/netInit.lua')
local vUtils = dofile('../utils/visUtils.lua')
local model_utils = dofile('../utils/model_utils.lua')
local tv=dofile('../nnutils/TotalVariation.lua')
-----------------------------
--------parameters-----------
local params = {}
--params.bgVal = 0
params.name = 'shapenetVoxels'
params.gpu = 1
params.batchSize = 8
params.imgSizeY = 64
params.imgSizeX = 64
params.synset = 2958343 --chair:3001627, aero:2691156, car:2958343

params.gridSizeX = 32
params.gridSizeY = 32
params.gridSizeZ = 32

params.lambda_l1 = 1
params.lambda_tv = 1e-6
params.drop_mask=12
params.matsave=1
params.imsave = 0
params.disp = 0
params.bottleneckSize = 800
params.noiseSize = 1
params.visIter = 100
params.nConvEncLayers = 5
params.nConvDecLayers = 4
params.nConvEncChannelsInit = 8
params.nVoxelChannels = 3
params.nOccChannels = 1
params.numTrainIter = 10000
params.ip = '131.159.40.120'
params.port = 8000
-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(params) do params[k] = tonumber(os.getenv(k)) or os.getenv(k) or params[k] end
if params.disp == 0 then params.display = false else params.display = true end
if params.imsave == 0 then params.imsave = false end
params.visDir = '../cachedir/visualization/' .. params.name
params.snapshotDir = '../cachedir/snapshots/shapenet/' .. params.name
params.imgSize = torch.Tensor({params.imgSizeX, params.imgSizeY})
params.gridSize = torch.Tensor({params.gridSizeX, params.gridSizeY, params.gridSizeZ})
params.synset = '0' .. tostring(params.synset) --to resolve string/number issues in passing bash arguments
--params.modelsDataDir = '../cachedir/blenderRenderPreprocess/' .. params.synset .. '/'
--params.modelsDataDir = '../../../arnab/nips16_PTN/data/shapenetcore_viewdata/' .. params.synset .. '/'
params.modelsDataDir='/home/viveka/'..params.synset .. '/'
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
--local colLossFunc = nn.AbsCriterion()           
local colLossFunc=nn.MSECriterion()
local ganLossFunc = nn.BCECriterion()
-----------------------------
----------Encoder-----------
local G={}
local nOutChannels=3
--G.encoder, nOutChannels = netBlocks.convEncoderSimple2d(params.nConvEncLayers,params.nConvEncChannelsInit,3,true) --output is nConvEncChannelsInit*pow(2,nConvEncLayers-1) X imgSize/pow(2,nConvEncLayers)
local featSpSize = params.imgSize/torch.pow(2,params.nConvEncLayers)
----print(featSpSize)
--local bottleneck = nn.Sequential():add(nn.Reshape(nOutChannels*featSpSize[1]*featSpSize[2],1,1,true))
--local nInputCh = nOutChannels*featSpSize[1]*featSpSize[2]
--for nLayers=1,2 do --fc for joint reasoning
--    bottleneck:add(nn.SpatialConvolution(nInputCh,params.bottleneckSize,1,1)):add(nn.SpatialBatchNormalization(params.bottleneckSize)):add(nn.LeakyReLU(0.2, true))
--    nInputCh = params.bottleneckSize
--end
--G.encoder:add(bottleneck)
--
G.encoder = netBlocks.convEncoderSimple3d(3,64,params.bottleneckSize,true)
G.encoder:add(nn.Reshape(params.bottleneckSize,1,1,true))
G.encoder:apply(netInit.weightsInit)
--print(G.encoder)
---------------------------------
----------World Decoder----------
local featSpSize = params.gridSize/torch.pow(2,params.nConvDecLayers)
G.decoder = nn.Sequential()
local parallel_inputs=nn.ParallelTable():add(nn.Identity()):add(nn.Identity())
G.decoder:add(parallel_inputs)
G.decoder:add(nn.JoinTable(1,3))
G.decoder:add(nn.SpatialConvolution(params.bottleneckSize + params.noiseSize,nOutChannels*featSpSize[1]*featSpSize[2]*featSpSize[3],1,1,1)):add(nn.SpatialBatchNormalization(nOutChannels*featSpSize[1]*featSpSize[2]*featSpSize[3])):add(nn.ReLU(true)):add(nn.Reshape(nOutChannels,featSpSize[1],featSpSize[2],featSpSize[3],true))
G.decoder:add(netBlocks.convDecoderSimple3d(params.nConvDecLayers,nOutChannels,params.nConvEncChannelsInit,params.nVoxelChannels,true))
local tv_mod=nn.TotalVariation(params.lambda_tv):cuda()
G.decoder:add(tv_mod)
G.decoder:apply(netInit.weightsInit)

local netD=netBlocks.SimpleDiscriminator(3,64,true)
netD:apply(netInit.weightsInit)
print(netD)
-----------------------------
----------Recons-------------
local splitUtil = dofile('../benchmark/synthetic/splits.lua')
local trainModels = splitUtil.getSplit(params.synset)['train']
local dataLoader = data.dataLoader(params.modelsDataDir, params.voxelsDir, params.batchSize, params.imgSize, params.gridSize, trainModels)
--local netRecons = nn.Sequential():add(G.encoder):add(G.decoder)
--local netRecons = torch.load(params.snapshotDir .. '/iter10000.t7')
--netRecons = netRecons:cuda()
for k,net in pairs(G) do net:cuda() end
netD=netD:cuda()
lossFunc = lossFunc:cuda()
colLossFunc = colLossFunc:cuda()
ganLossFunc = ganLossFunc:cuda()
--print(G.encoder)
--print(decoder)
local err = 0
local errD = 0
-- Optimization parameters
local optimState = {
   learningRate = 0.0001,
   beta1 = 0.9,
}
local optimStateD = {
   learningRate = 0.0001,
   beta1 = 0.9,
}
--local netParameters, netGradParameters = netRecons:getParameters()
local netParameters, netGradParameters = model_utils.combine_all_parameters(G)
local netDParameters, netDGradParameters = netD:getParameters()
local tm = torch.Timer()
local data_tm = torch.Timer()
local imgs, pred, rays
-- fX required for training
local input=torch.Tensor(params.batchSize,3,params.gridSizeX,params.gridSizeY,params.gridSizeZ)
local label = torch.Tensor(params.batchSize)
input=input:cuda()
label=label:cuda()
local real_label=1
local fake_label=0
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

    voxelsGt:cmul(occMask)


    imgs = imgs:cuda()
    voxelsGt = voxelsGt:cuda()
    voxelsOcc= voxelsOcc:cuda()
    occMask=occMask:cuda()


    local impaintMask=occMask:clone()
    local xStart= params.gridSizeX/2-params.drop_mask/2 --torch.random(1,params.gridSizeX-params.drop_mask)
    local yStart= params.gridSizeY/2-params.drop_mask/2 -- torch.random(1,params.gridSizeY-params.drop_mask)
    local zStart= params.gridSizeZ/2-params.drop_mask/2 -- torch.random(1,params.gridSizeZ-params.drop_mask)

    impaintMask[ { { } , { } , { xStart , xStart + params.drop_mask } , { yStart,yStart+params.drop_mask } , { zStart , zStart+params.drop_mask }    } ]:fill(0)

    droppedVoxelsGt=torch.cmul(voxelsGt,impaintMask)

    local encoded=G.encoder:forward(droppedVoxelsGt)
    local noise=torch.Tensor(params.batchSize,params.noiseSize,1,1):cuda()
    noise:normal(0,1)
    G.decoder:forward({encoded,noise})

    color=G.decoder.output

    color:cmul(occMask)
 
    --err = lossFunc:forward(pred, voxelsOcc)
    err =params.lambda_l1*colLossFunc:forward(color,voxelsGt)
    --local gradPred = lossFunc:backward(pred, voxelsOcc):mul(params.lambda_l1)
    local gradColor = colLossFunc:backward(color,voxelsGt):mul(params.lambda_l1)
    label:fill(real_label)
    local output=netD:forward(color)
    err = err + ganLossFunc:forward(output,label)
    local df_do= ganLossFunc:backward(output,label)
    gradColor = netD:updateGradInput(color,df_do) + gradColor -- trying just adversarial loss

    gradColor:cmul(occMask)

    local d_decoder = G.decoder:backward({encoded,noise}, gradColor )
    G.encoder:backward( droppedVoxelsGt ,d_decoder[1])
    tm:stop()
    return err, netGradParameters
end

local fDx = function(x)
    netDGradParameters:zero()

    -- train with real colored voxels
    input:copy(voxelsGt)
    label:fill(real_label)
    local output=netD:forward(input)
    local errD_real=ganLossFunc:forward(output,label)
    local df_do = ganLossFunc:backward(output,label)
    netD:backward(input,df_do)

    -- train with generated colored voxels
    input:copy(color)
    label:fill(fake_label)
    local output=netD:forward(input)
    local errD_fake=ganLossFunc:forward(output,label)
    local df_do = ganLossFunc:backward(output,label)
    netD:backward(input,df_do)
    
    errD= errD_real + errD_fake
    return errD, netDGradParameters
end

--print(netRecons)
-----------------------------
----------Training-----------
if(params.display) then 
    disp = require 'display' 
    disp.configure({hostname=params.ip, port=params.port})
end
local forwIter = 0
for iter=1,params.numTrainIter do
   
    print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
        .. '  Err_G: %.4f  Err_D: %.4f'):format(
        iter, ((iter-1) / params.batchSize),
        math.floor( params.numTrainIter / params.batchSize),
        tm:time().real, data_tm:time().real,
        err and err or -1, errD and errD or -1))
    --print(('Data/Total time : %f/%f'):format(data_tm:time().real,tm:time().real))
    fout:write(string.format('%d %f\n',iter,err))
    fout:flush()
    if(iter%params.visIter==0) then
        local dispVar = color:clone()  --pred:clone()
        if(params.disp == 1) then
            disp.image(imgs, {win=10, title='inputIm'})
            disp.image(dispVar:max(3):squeeze(), {win=1, title='predX'})
            disp.image(dispVar:max(4):squeeze(), {win=2, title='predY'})
            disp.image(dispVar:max(5):squeeze(), {win=3, title='predZ'})
            
            disp.image(voxelsGt:max(3):squeeze(), {win=11, title='gtX'})
            disp.image(voxelsGt:max(4):squeeze(), {win=12, title='gtY'})
            disp.image(voxelsGt:max(5):squeeze(), {win=13, title='gtZ'})

            disp.image(droppedVoxelsGt:max(3):squeeze(), {win=21, title='droppedX'})
            disp.image(droppedVoxelsGt:max(4):squeeze(), {win=22, title='droppedY'})
            disp.image(droppedVoxelsGt:max(5):squeeze(), {win=23, title='droppedZ'})

        end
        if(params.imsave == 1) then
            vUtils.imsave(imgs, params.visDir .. '/inputIm'.. iter .. '.png')
            vUtils.imsave(dispVar:max(3):squeeze(), params.visDir.. '/predX' .. iter .. '.png')
            vUtils.imsave(dispVar:max(4):squeeze(), params.visDir.. '/predY' .. iter .. '.png')
            vUtils.imsave(dispVar:max(5):squeeze(), params.visDir.. '/predZ' .. iter .. '.png')
        end
        if(params.matsave==1) then
            local vox_dir=params.voxelSaveDir.. tostring(iter)
            paths.mkdir(vox_dir)
            for i =1,params.batchSize do
                matio.save(vox_dir .. string.format('/gt_%03d.mat',i),voxelsGt[i]:float())
                matio.save(vox_dir ..  string.format('/pred_%03d.mat',i),color[i]:float())
                --matio.save(vox_dir ..  string.format('/pred_occ%03d.mat',i),pred[i]:float())
            end
        end
    end
    if(iter%5000)==0 then
        torch.save(params.snapshotDir .. '/iter'.. iter .. '_netG.t7', {G=G})
        torch.save(params.snapshotDir .. '/iter'.. iter .. '_NetD.t7', netD)
    end
    optim.adam(fx, netParameters, optimState)
    optim.adam(fDx,netDParameters,optimStateD)
end
