torch.manualSeed(1)
require 'cunn'
require 'optim'
matio=require 'matio'
local data = dofile('../data/synthetic/shapenetColorVoxels.lua')
--local data = dofile('../data/synthetic/shapenetColorRenderedVoxels.lua')
local netBlocks = dofile('../nnutils/netBlocks.lua')
local netInit = dofile('../nnutils/netInit.lua')
local vUtils = dofile('../utils/visUtils.lua')
local model_utils = dofile('../utils/model_utils.lua')
-----------------------------
--------parameters-----------
local params = {}
--params.bgVal = 0
params.name = 'shapenetVoxels'
params.gpu = 1
params.batchSize = 32
params.imgSizeY = 64
params.imgSizeX = 64
params.synset = 3001627 --2958343 --chair:3001627, aero:2691156, car:2958343

params.gridSizeX = 32
params.gridSizeY = 32
params.gridSizeZ = 32
params.lambda=0.9
params.lambda_gan=1
params.matsave=1
params.imsave = 0
params.disp = 0
params.bottleneckSize = 400
params.noiseSize=1
params.visIter = 1000
params.nConvEncLayers = 5
params.nConvDecLayers = 4
params.nConvEncChannelsInit = 8
params.nVoxelChannels = 3
params.nOccChannels = 1
params.numTrainIter = 500000
params.ip = '129.67.94.233'--'131.159.40.120'
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
--params.modelsDataDir = '../../../arnab/nips16_PTN/data/shapenetcore_viewdata/' .. params.synset .. '/'
--params.modelsDataDir='/mnt/raid/viveka/data/'..params.synset .. '/'
params.modelsDataDir='../../data/color-3d/Images/'..params.synset .. '/'
params.voxelsDir = '../../data/color-3d/vox_dim32/' .. params.synset .. '/'
--params.voxelsDir = '../cachedir/shapenet/modelVoxels/' .. params.synset .. '/'
--params.voxelsDir = '../../../arnab/nips16_PTN/data/shapenetcore_colvoxdata/' .. params.synset .. '/'
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
local occ_ganLossFunc = nn.BCECriterion()
local col_ganLossFunc = nn.BCECriterion()
-----------------------------
----------------------------------------
local G={}
local nOutChannels=nil
G.encoder, nOutChannels = netBlocks.convEncoderSimple2d(params.nConvEncLayers,params.nConvEncChannelsInit,3,true) --output is nConvEncChannelsInit*pow(2,nConvEncLayers-1) X imgSize/pow(2,nConvEncLayers)
local featSpSize = params.imgSize/torch.pow(2,params.nConvEncLayers)
--print(featSpSize)
local bottleneck = nn.Sequential():add(nn.Reshape(nOutChannels*featSpSize[1]*featSpSize[2],1,1,true))
local nInputCh = nOutChannels*featSpSize[1]*featSpSize[2]
for nLayers=1,2 do --fc for joint reasoning
    bottleneck:add(nn.SpatialConvolution(nInputCh,params.bottleneckSize,1,1)):add(nn.SpatialBatchNormalization(params.bottleneckSize)):add(nn.LeakyReLU(0.2, true))
    nInputCh = params.bottleneckSize
end
G.encoder:add(bottleneck)
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
G.decoder:add(netBlocks.convDecoderSimple3dHeads(params.nConvDecLayers,nOutChannels,params.nConvEncChannelsInit,params.nVoxelChannels,params.nOccChannels,true))
G.decoder:apply(netInit.weightsInit)


local occ_netD=netBlocks.ConditionalDiscriminator(1,3,64,true)
local col_netD=netBlocks.ConditionalDiscriminator(3,3,64,true)
-----------------------------
----------Recons-------------
local splitUtil = dofile('../benchmark/synthetic/splits.lua')
local trainModels = splitUtil.getSplit(params.synset)['train']
local testModels = splitUtil.getSplit(params.synset)['test']
--local trainModels = {trainModels[1]}
--print(trainModels)
local dataLoader = data.dataLoader(params.modelsDataDir, params.voxelsDir, params.batchSize, params.imgSize, params.gridSize, trainModels)
local dataLoaderTest = data.dataLoader(params.modelsDataDir, params.voxelsDir, params.batchSize, params.imgSize, params.gridSize, testModels)

for k,net in pairs(G) do net:cuda() end
occ_netD=occ_netD:cuda()
col_netD=col_netD:cuda()
lossFunc = lossFunc:cuda()
colLossFunc = colLossFunc:cuda()
occ_ganLossFunc =occ_ganLossFunc:cuda()
col_ganLossFunc =col_ganLossFunc:cuda()
--print(encoder)
--print(decoder)
local err = 0
local occ_errD=0
local col_errD=0
-- Optimization parameters
local optimState = {
   learningRate = 0.0001,
   beta1 = 0.9,
}
local occ_optimState = {
   learningRate = 0.0001,
   beta1 = 0.9,
}
local col_optimState = {
   learningRate = 0.0001,
   beta1 = 0.9,
}

local netParameters, netGradParameters = model_utils.combine_all_parameters(G)
local occ_netDParameters, occ_netDGradParameters = occ_netD:getParameters()
local col_netDParameters, col_netDGradParameters = col_netD:getParameters()
local tm = torch.Timer()
local data_tm = torch.Timer()
local imgs, pred, rays,voxelsOcc, voxelsGt 

local col_input=torch.Tensor(params.batchSize,3,params.gridSizeX,params.gridSizeY,params.gridSizeZ)
local occ_input=torch.Tensor(params.batchSize,1,params.gridSizeX,params.gridSizeY,params.gridSizeZ)
local label = torch.Tensor(params.batchSize)
col_input=col_input:cuda()
occ_input=occ_input:cuda()
label=label:cuda()
local real_label=1
local fake_label=0

-- fX required for training
local fx = function(x)
    tm:reset(); tm:resume()
    netGradParameters:zero()
    data_tm:reset(); data_tm:resume()
    imgs, voxelsGt = dataLoader:forward()
    data_tm:stop()
    err=0
    --print('Data loaded')
    voxelsOcc=torch.sum(voxelsGt,2)
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

    local encoded=G.encoder:forward(imgs)
    local noise=torch.Tensor(params.batchSize,params.noiseSize,1,1):cuda()
    noise:normal(0,1)
    G.decoder:forward({encoded,noise})

    color=G.decoder.output[1]
    pred=G.decoder.output[2]
   
    color:cmul(occMask)
    voxelsGt:cmul(occMask)

    ----------------------------
    -- Play of the 2 gans
    ---------------------------
    label:fill(real_label)
    occ_output=occ_netD:forward({ pred,imgs})
    col_output=col_netD:forward({ color,imgs})

    err = err + (1-params.lambda)*(params.lambda_gan)*occ_ganLossFunc:forward(occ_output,label)
    err = err + (params.lambda)*(params.lambda_gan)*col_ganLossFunc:forward(col_output,label)

    local occ_df_do=occ_ganLossFunc:backward(occ_output,label)
    local col_df_do=col_ganLossFunc:backward(col_output,label)
    --------------------------------

    err = err + (1-params.lambda)*(1-params.lambda_gan)*lossFunc:forward(pred, voxelsOcc)
    err = err + (params.lambda)*(1-params.lambda_gan)*colLossFunc:forward(color,voxelsGt)
    local gradPred = (1-params.lambda)*(1-params.lambda_gan)*lossFunc:backward(pred, voxelsOcc)
    gradPred=gradPred+(1-params.lambda)*(params.lambda_gan)*occ_netD:updateGradInput({pred,imgs},occ_df_do)[1]
    local gradColor = (params.lambda)*(1-params.lambda_gan)*colLossFunc:backward(color,voxelsGt)
    gradColor=gradColor + (params.lambda)*(params.lambda_gan)*col_netD:updateGradInput({color,imgs},col_df_do)[1]

    local d_decoder = G.decoder:backward({encoded,noise}, { gradColor , gradPred})
    G.encoder:backward(imgs,d_decoder[1])

    tm:stop()
    return err, netGradParameters
end

local occ_fDx = function(x)
    occ_netDGradParameters:zero()

    -- train with real colored voxels
    occ_input:copy(voxelsOcc)
    label:fill(real_label)
    local output=occ_netD:forward({occ_input,imgs})
    local errD_real=occ_ganLossFunc:forward(output,label)
    local df_do = occ_ganLossFunc:backward(output,label)
    occ_netD:backward({occ_input,imgs},df_do)

    -- train with generated occored voxels
    occ_input:copy(pred)
    label:fill(fake_label)
    local output=occ_netD:forward({occ_input,imgs})
    local errD_fake=occ_ganLossFunc:forward(output,label)
    local df_do = occ_ganLossFunc:backward(output,label)
    occ_netD:backward({occ_input,imgs},df_do)
    
    occ_errD = errD_real + errD_fake
    return occ_errD, occ_netDGradParameters
end



local col_fDx = function(x)
    col_netDGradParameters:zero()

    -- train with real colored voxels
    col_input:copy(voxelsGt)
    label:fill(real_label)
    local output=col_netD:forward({col_input,imgs})
    local errD_real=col_ganLossFunc:forward(output,label)
    local df_do = col_ganLossFunc:backward(output,label)
    col_netD:backward({col_input,imgs},df_do)

    -- train with generated colored voxels
    col_input:copy(color)
    label:fill(fake_label)
    local output=col_netD:forward({col_input,imgs})
    local errD_fake=col_ganLossFunc:forward(output,label)
    local df_do = col_ganLossFunc:backward(output,label)
    col_netD:backward({col_input,imgs},df_do)
    
    col_errD = errD_real + errD_fake
    return col_errD, col_netDGradParameters
end

---------------New Eval
-----------------------
local err_test=0
function eval()
    G.encoder:evaluate()
    G.decoder:evaluate()
    imgs, voxelsGt = dataLoaderTest:forward()
    err_test=0
    --print('Data loaded')
    voxelsOcc=torch.sum(voxelsGt,2)
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

    local encoded=G.encoder:forward(imgs)
    local noise=torch.Tensor(params.batchSize,params.noiseSize,1,1):cuda()
    noise:normal(0,1)
    G.decoder:forward({encoded,noise})

    color=G.decoder.output[1]
    pred=G.decoder.output[2]
   
    color:cmul(occMask)
    voxelsGt:cmul(occMask)

    ----------------------------
    -- Play of the 2 gans
    ---------------------------
    label:fill(real_label)
    occ_output=occ_netD:forward({ pred,imgs})
    col_output=col_netD:forward({ color,imgs})

    err_test = err_test + (1-params.lambda)*(params.lambda_gan)*occ_ganLossFunc:forward(occ_output,label)
    err_test = err_test + (params.lambda)*(params.lambda_gan)*col_ganLossFunc:forward(col_output,label)

    --------------------------------

    err_test = err_test + (1-params.lambda)*(1-params.lambda_gan)*lossFunc:forward(pred, voxelsOcc)
    err_test = err_test + (params.lambda)*(1-params.lambda_gan)*colLossFunc:forward(color,voxelsGt)
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
    --print(iter,err)
    --print(('Data/Total time : %f/%f'):format(data_tm:time().real,tm:time().real))
    --fout:write(string.format('%d %f\n',iter,err))
    print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
        .. '  Err_G: %.4f  Err_occ_D: %.4f Err_col_D: %.4f '):format(
        iter, ((iter-1) / params.batchSize),
        math.floor( params.numTrainIter / params.batchSize),
        tm:time().real, data_tm:time().real,
        err and err or -1, occ_errD and occ_errD or -1,col_errD and col_errD or -1))

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
        torch.save(params.snapshotDir .. '/iter'.. iter .. '_netG.t7', {G=G} )
        torch.save(params.snapshotDir .. '/iter'.. iter .. '_occD.t7', occ_netD )
        torch.save(params.snapshotDir .. '/iter'.. iter .. '_colG.t7', col_netD )
    end
    optim.adam(fx, netParameters, optimState)
    optim.adam(col_fDx, col_netDParameters, col_optimState)
    optim.adam(occ_fDx, occ_netDParameters, occ_optimState)
end
