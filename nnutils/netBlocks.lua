require 'nn'
require 'nngraph'
local M = {}
local tv=dofile('../nnutils/TotalVariation.lua')
function M.convEncoderSimple1d(nLayers, nChannelsInit, nInputChannels, useBn)
    local nInputChannels = nInputChannels or 3
    local nChannelsInit = nChannelsInit or 8
    local useBn = useBn~=false and true
    local nOutputChannels = nChannelsInit
    local encoder = nn.Sequential()
    
    for l=1,nLayers do
        encoder:add(nn.TemporalConvolution(nInputChannels, nOutputChannels, 3, 1))
        encoder:add(nn.LeakyReLU(0.2, true))
        encoder:add(nn.TemporalMaxPooling(2, 2))
        nInputChannels = nOutputChannels
        nOutputChannels = nOutputChannels*2
    end
    return encoder, nOutputChannels/2 -- division by two offsets the mutiplication in last iteration
end

function M.convEncoderSimple2d(nLayers, nChannelsInit, nInputChannels, useBn)
    local nInputChannels = nInputChannels or 3
    local nChannelsInit = nChannelsInit or 8
    local useBn = useBn~=false and true
    local nOutputChannels = nChannelsInit
    local encoder = nn.Sequential()
    
    for l=1,nLayers do
        encoder:add(nn.SpatialConvolution(nInputChannels, nOutputChannels, 3, 3, 1, 1, 1, 1))
        if useBn then encoder:add(nn.SpatialBatchNormalization(nOutputChannels)) end
        encoder:add(nn.LeakyReLU(0.2, true))
        encoder:add(nn.SpatialMaxPooling(2, 2, 2, 2))
        nInputChannels = nOutputChannels
        nOutputChannels = nOutputChannels*2
    end
    return encoder, nOutputChannels/2 -- division by two offsets the mutiplication in last iteration
end

function M.convEncoderComplex2d(nLayers, nChannelsInit, nInputChannels, useBn)
    local nInputChannels = nInputChannels or 3
    local nChannelsInit = nChannelsInit or 8
    local useBn = useBn~=false and true
    local nOutputChannels = nChannelsInit
    local encoder = nn.Sequential()
    
    for l=1,nLayers do
        encoder:add(nn.SpatialConvolution(nInputChannels, nOutputChannels, 3, 3, 1, 1, 1, 1))
        if useBn then encoder:add(nn.SpatialBatchNormalization(nOutputChannels)) end
        encoder:add(nn.LeakyReLU(0.2, true))
        
        encoder:add(nn.SpatialConvolution(nOutputChannels, nOutputChannels, 3, 3, 1, 1, 1, 1))
        if useBn then encoder:add(nn.SpatialBatchNormalization(nOutputChannels)) end
        encoder:add(nn.LeakyReLU(0.2, true))
        
        encoder:add(nn.SpatialMaxPooling(2, 2, 2, 2))
        nInputChannels = nOutputChannels
        nOutputChannels = nOutputChannels*2
    end
    return encoder, nOutputChannels/2 -- division by two offsets the mutiplication in last iteration
end

function M.convEncoderSimple3d(nInputChannels,ndf, bottleneckSize, useBn)
   local netD = nn.Sequential()
   local useBn = useBn ~= false and true
   local ndf = ndf or 8
   local bottleneckSize = bottleneckSize or 100

   --input is (nInputChannels) x 32 x 32 x 32
   netD:add(nn.VolumetricConvolution(nInputChannels, ndf, 4, 4, 4, 2, 2, 2, 1, 1, 1))
   netD:add(nn.LeakyReLU(0.2, true))
   -- state size: (ndf) x 16 x 16 x 16
   netD:add(nn.VolumetricConvolution(ndf, ndf * 2, 4, 4, 4 , 2, 2, 2 ,1 , 1, 1))
   netD:add(nn.VolumetricBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
   -- state size: (ndf*2) x 8 x 8 x 8
   netD:add(nn.VolumetricConvolution(ndf * 2, ndf * 4, 4, 4, 4, 2, 2, 2,1, 1, 1))
   netD:add(nn.VolumetricBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
   -- state size: (ndf*4) x 4 x 4 x 4
   --netD:add(VolumetricConvolution(ndf * 4, ndf * 8, 4, 4, 4, 2, 2,2,1, 1,1))
   --netD:add(SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
   -- state size: (ndf*8) x 2 x 2
   netD:add(nn.VolumetricConvolution(ndf * 4, bottleneckSize , 4, 4 , 4))
   netD:add(nn.LeakyReLU(0.2,true))
   -- state size: 1 x 1 x 1
   --netD:add(nn.View(1):setNumInputDims(3))
   -- state size: 1
   return netD
end


function M.convDecoderSimple2d(nLayers, nInputChannels, ndf, nFinalChannels, useBn)
    --adds nLayers deconv layers + 1 conv layer
    local nFinalChannels = nFinalChannels or 3
    local ndf = ndf or 8 --channels in penultimate layer
    local useBn = useBn~=false and true
    local nOutputChannels = ndf*torch.pow(2,nLayers-1)
    local decoder = nn.Sequential()
    for l=1,nLayers do
        decoder:add(nn.SpatialFullConvolution(nInputChannels, nOutputChannels, 4, 4, 2, 2, 1, 1))
        if useBn then decoder:add(nn.SpatialBatchNormalization(nOutputChannels)) end
        decoder:add(nn.ReLU(true))
        nInputChannels = nOutputChannels
        nOutputChannels = nOutputChannels/2
    end
    decoder:add(nn.SpatialConvolution(ndf, nFinalChannels, 3, 3, 1, 1, 1, 1))
    decoder:add(nn.Tanh()):add(nn.AddConstant(1)):add(nn.MulConstant(0.5))
    return decoder
end

function M.convDecoderComplex2d(nLayers, nInputChannels, ndf, nFinalChannels, useBn)
    --adds nLayers deconv-conv layers + 1 final conv layer
    local nFinalChannels = nFinalChannels or 3
    local ndf = ndf or 8 --channels in penultimate layer
    local useBn = useBn~=false and true
    local nOutputChannels = ndf*torch.pow(2,nLayers-1)
    local decoder = nn.Sequential()
    for l=1,nLayers do
        decoder:add(nn.SpatialFullConvolution(nInputChannels, nOutputChannels, 4, 4, 2, 2, 1, 1))
        if useBn then decoder:add(nn.SpatialBatchNormalization(nOutputChannels)) end
        decoder:add(nn.LeakyReLU(0.2, true))
        
        decoder:add(nn.SpatialConvolution(nOutputChannels, nOutputChannels, 3, 3, 1, 1, 1, 1))
        if useBn then decoder:add(nn.SpatialBatchNormalization(nOutputChannels)) end
        decoder:add(nn.ReLU(true))
        nInputChannels = nOutputChannels
        nOutputChannels = nOutputChannels/2
    end
    decoder:add(nn.SpatialConvolution(ndf, nFinalChannels, 3, 3, 1, 1, 1, 1))
    decoder:add(nn.Tanh()):add(nn.AddConstant(1)):add(nn.MulConstant(0.5))
    return decoder
end

function M.convDecoderSimple3d(nLayers, nInputChannels, ndf, nFinalChannels, useBn, normalizeOut)
    --adds nLayers deconv layers + 1 conv layer
    local nFinalChannels = nFinalChannels or 1
    local ndf = ndf or 8 --channels in penultimate layer
    local useBn = useBn~=false and true
    local normalizeOut = normalizeOut~=false and true
    local nOutputChannels = ndf*torch.pow(2,nLayers-1)
    local decoder = nn.Sequential()
    for l=1,nLayers do
        decoder:add(nn.VolumetricFullConvolution(nInputChannels, nOutputChannels, 4, 4, 4, 2, 2, 2, 1, 1, 1))
        if useBn then decoder:add(nn.VolumetricBatchNormalization(nOutputChannels)) end
        decoder:add(nn.ReLU(true))
        nInputChannels = nOutputChannels
        nOutputChannels = nOutputChannels/2
    end
    decoder:add(nn.VolumetricConvolution(ndf, nFinalChannels, 3, 3, 3, 1, 1, 1, 1, 1, 1))
    if(normalizeOut) then
        decoder:add(nn.Tanh()):add(nn.AddConstant(1)):add(nn.MulConstant(0.5))
    end
    return decoder
end

function M.convDecoderSimple3dHeads(nLayers, nInputChannels, ndf, nFinalChannels1, nFinalChannels2, useBn, normalizeOut,useTV,lambda_tv)
    --adds nLayers deconv layers + 1 conv layer
    local nFinalChannels = nFinalChannels or 1
    local ndf = ndf or 8 --channels in penultimate layer
    local useBn = useBn~=false and true
    local normalizeOut = normalizeOut~=false and true
    local nOutputChannels = ndf*torch.pow(2,nLayers-1)
    local decoder = nn.Sequential()
    local lambda_tv = lambda_tv or 1e-6
    local useTV = useTV~=false and true
    for l=1,nLayers do
        decoder:add(nn.VolumetricFullConvolution(nInputChannels, nOutputChannels, 4, 4, 4, 2, 2, 2, 1, 1, 1))
        if useBn then decoder:add(nn.VolumetricBatchNormalization(nOutputChannels)) end
        decoder:add(nn.ReLU(true))
        nInputChannels = nOutputChannels
        nOutputChannels = nOutputChannels/2
    end
    local head1=nn.Sequential()
    local head2=nn.Sequential()
    head1:add(nn.VolumetricConvolution(ndf, nFinalChannels1, 3, 3, 3, 1, 1, 1, 1, 1, 1))
    head2:add(nn.VolumetricConvolution(ndf, nFinalChannels2, 3, 3, 3, 1, 1, 1, 1, 1, 1))

    if(normalizeOut) then
        head1:add(nn.Tanh()):add(nn.AddConstant(1)):add(nn.MulConstant(0.5))
        head2:add(nn.Tanh()):add(nn.AddConstant(1)):add(nn.MulConstant(0.5))
    end
    if useTV then
        local tv_mod=nn.TotalVariation(lambda_tv):cuda()
        head2:add(tv_mod)
    end
    decoder:add(nn.ConcatTable():add(head1):add(head2))
    return decoder
end

function M.SimpleDiscriminator(nInputChannels,ndf,useBn)
   local netD = nn.Sequential()
   local useBn = useBn ~= false and true
   local ndf = ndf or 8

   --input is (nInputChannels) x 32 x 32 x 32
   netD:add(nn.VolumetricConvolution(nInputChannels, ndf, 4, 4, 4, 2, 2, 2, 1, 1, 1))
   netD:add(nn.LeakyReLU(0.2, true))
   -- state size: (ndf) x 16 x 16 x 16
   netD:add(nn.VolumetricConvolution(ndf, ndf * 2, 4, 4, 4 , 2, 2, 2 ,1 , 1, 1))
   netD:add(nn.VolumetricBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
   -- state size: (ndf*2) x 8 x 8 x 8
   netD:add(nn.VolumetricConvolution(ndf * 2, ndf * 4, 4, 4, 4, 2, 2, 2,1, 1, 1))
   netD:add(nn.VolumetricBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
   -- state size: (ndf*4) x 4 x 4 x 4
   --netD:add(VolumetricConvolution(ndf * 4, ndf * 8, 4, 4, 4, 2, 2,2,1, 1,1))
   --netD:add(SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
   -- state size: (ndf*8) x 2 x 2
   netD:add(nn.VolumetricConvolution(ndf * 4, 1, 4, 4 , 4))
   netD:add(nn.Sigmoid())
   -- state size: 1 x 1 x 1
   netD:add(nn.View(1):setNumInputDims(3))
   -- state size: 1
   return netD
end

function M.ConditionalDiscriminator(nInputChannels,ndf,useBn)
   local useBn = useBn ~= false and true
   local ndf = ndf or 8

   local netD=nn.Sequential()
   local net3D = nn.Sequential()
   --input is (nInputChannels) x 32 x 32 x 32
   net3D:add(nn.VolumetricConvolution(nInputChannels, ndf, 4, 4, 4, 2, 2, 2, 1, 1, 1))
   net3D:add(nn.LeakyReLU(0.2, true))
   -- state size: (ndf) x 16 x 16 x 16
   net3D:add(nn.VolumetricConvolution(ndf, ndf * 2, 4, 4, 4 , 2, 2, 2 ,1 , 1, 1))
   net3D:add(nn.VolumetricBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
   -- state size: (ndf*2) x 8 x 8 x 8
   net3D:add(nn.VolumetricConvolution(ndf * 2, ndf * 4, 4, 4, 4, 2, 2, 2,1, 1, 1))
   net3D:add(nn.VolumetricBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
   -- state size: (ndf*4) x 4 x 4 x 4
   net3D:add(nn.Reshape( ndf*4*4*4*4))

   local net2D = nn.Sequential()
   --input is (nInputChannels) x 64 x 64
   net2D:add(nn.SpatialConvolution(nInputChannels, ndf, 4, 4,  2, 2,  1, 1))
   net2D:add(nn.LeakyReLU(0.2, true))
   -- state size: (ndf) x 32 x 32 
   net2D:add(nn.SpatialConvolution(ndf, ndf * 2, 4, 4 , 2,  2 ,1 ,  1))
   net2D:add(nn.SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
   -- state size: (ndf*2) x 16 x 16 
   net2D:add(nn.SpatialConvolution(ndf * 2, ndf * 4,  4, 4, 2, 2,1, 1))
   net2D:add(nn.SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))

   -- state size: (ndf*4) x 8 x 8
   net2D:add(nn.SpatialConvolution(ndf *4, ndf * 4,  4, 4, 2, 2,1, 1))
   net2D:add(nn.SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
 
   -- state size: (ndf*4) x 4 x 4
   net2D:add(nn.Reshape(ndf*4*4*4))
    
   netD:add(nn.ParallelTable():add(net3D):add(net2D))
   netD:add(nn.JoinTable(1,1))
   netD:add(nn.Linear(ndf*4*4*4*5, ndf*4*4*4)) 
   netD:add(nn.BatchNormalization(ndf*4*4*4)):add(nn.LeakyReLU(0.2,true))
   
   netD:add(nn.Linear(ndf*4*4*4, ndf*4*4)) 
   netD:add(nn.BatchNormalization(ndf*4*4)):add(nn.LeakyReLU(0.2,true))

   netD:add(nn.Linear(ndf*4*4, ndf*4)) 
   netD:add(nn.BatchNormalization(ndf*4)):add(nn.LeakyReLU(0.2,true))

   netD:add(nn.Linear(ndf*4, 1))

   netD:add(nn.Sigmoid())

   return netD
end

function M.ImageOccupancyEncoder(nInputChannels2D,nInputChannels3D,ndf, bottleneckSize , useBn)
   local useBn = useBn ~= false and true
   local ndf = ndf or 8

   local netD=nn.Sequential()
   local net3D = nn.Sequential()
   --input is (nInputChannels) x 32 x 32 x 32
   net3D:add(nn.VolumetricConvolution(nInputChannels3D, ndf, 4, 4, 4, 2, 2, 2, 1, 1, 1))
   net3D:add(nn.LeakyReLU(0.2, true))
   -- state size: (ndf) x 16 x 16 x 16
   net3D:add(nn.VolumetricConvolution(ndf, ndf * 2, 4, 4, 4 , 2, 2, 2 ,1 , 1, 1))
   net3D:add(nn.VolumetricBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
   -- state size: (ndf*2) x 8 x 8 x 8
   net3D:add(nn.VolumetricConvolution(ndf * 2, ndf * 4, 4, 4, 4, 2, 2, 2,1, 1, 1))
   net3D:add(nn.VolumetricBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
   -- state size: (ndf*4) x 4 x 4 x 4
   net3D:add(nn.Reshape( ndf*4*4*4*4))

   local net2D = nn.Sequential()
   --input is (nInputChannels) x 64 x 64
   net2D:add(nn.SpatialConvolution(nInputChannels2D, ndf, 4, 4,  2, 2,  1, 1))
   net2D:add(nn.LeakyReLU(0.2, true))
   -- state size: (ndf) x 32 x 32 
   net2D:add(nn.SpatialConvolution(ndf, ndf * 2, 4, 4 , 2,  2 ,1 ,  1))
   net2D:add(nn.SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
   -- state size: (ndf*2) x 16 x 16 
   net2D:add(nn.SpatialConvolution(ndf * 2, ndf * 4,  4, 4, 2, 2,1, 1))
   net2D:add(nn.SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))

   -- state size: (ndf*4) x 8 x 8
   net2D:add(nn.SpatialConvolution(ndf *4, ndf * 4,  4, 4, 2, 2,1, 1))
   net2D:add(nn.SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
 
   -- state size: (ndf*4) x 4 x 4
   net2D:add(nn.Reshape(ndf*4*4*4))
    
   netD:add(nn.ParallelTable():add(net3D):add(net2D))
   netD:add(nn.JoinTable(1,1))
   netD:add(nn.Linear(ndf*4*4*4*5, ndf*4*4*4)) 
   netD:add(nn.BatchNormalization(ndf*4*4*4)):add(nn.LeakyReLU(0.2,true))
   
   netD:add(nn.Linear(ndf*4*4*4, ndf*4*4)) 
   netD:add(nn.BatchNormalization(ndf*4*4)):add(nn.LeakyReLU(0.2,true))

   netD:add(nn.Linear(ndf*4*4, bottleneckSize)) 
   netD:add(nn.BatchNormalization(bottleneckSize)):add(nn.LeakyReLU(0.2,true))

   --netD:add(nn.Linear(ndf*4, 1))

   --netD:add(nn.Sigmoid())

   return netD
end


function M.TwoDImageGenerator(nz,nc,ngf)
    local nz=nz or 100
    local nc = nc or 3
	local ngf=ngf or 64
	local netG = nn.Sequential()
	-- input is Z, going into a convolution
	netG:add(nn.SpatialFullConvolution(nz, ngf * 8, 4, 4))
	netG:add(nn.SpatialBatchNormalization(ngf * 8)):add(nn.ReLU(true))
	-- state size: (ngf*8) x 4 x 4
	netG:add(nn.SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))
	netG:add(nn.SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))
	-- state size: (ngf*4) x 8 x 8
	netG:add(nn.SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))
	netG:add(nn.SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))
	-- state size: (ngf*2) x 16 x 16
	netG:add(nn.SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1))
	netG:add(nn.SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
	-- state size: (ngf) x 32 x 32
	netG:add(nn.SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1))
	netG:add(nn.Tanh())
	-- state size: (nc) x 64 x 64
	return netG
end

function M.TwoDImageDiscriminator(nc,ndf)
    local nc = nc or 3
	local ndf = ndf or 64
	local netD = nn.Sequential()
	
	-- input is (nc) x 64 x 64
	netD:add(nn.SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))
	netD:add(nn.LeakyReLU(0.2, true))
	-- state size: (ndf) x 32 x 32
	netD:add(nn.SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
	netD:add(nn.SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
	-- state size: (ndf*2) x 16 x 16
	netD:add(nn.SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
	netD:add(nn.SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
	-- state size: (ndf*4) x 8 x 8
	netD:add(nn.SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
	netD:add(nn.SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
	-- state size: (ndf*8) x 4 x 4
	netD:add(nn.SpatialConvolution(ndf * 8, 1, 4, 4))
	netD:add(nn.Sigmoid())
	-- state size: 1 x 1 x 1
	netD:add(nn.View(1):setNumInputDims(3))
	-- state size: 1
    return netD	
end

function M.ThreeDColorVoxelGridEncoder(nz,nc,ndf)
    local nz=nz or 100
    local nc=nc or 3
    local ndf=ndf or 64

    local netD= nn.Sequential()
	-- input is (nc) x 32 x 32 x 32
	netD:add(nn.VolumetricConvolution(nc, ndf, 4, 4,4,2, 2, 2,1, 1, 1))
	netD:add(nn.LeakyReLU(0.2, true))
	-- state size: (ndf) x 16 x 16 x 16
	netD:add(nn.VolumetricConvolution(ndf, ndf * 2, 4, 4,4,2, 2, 2,1, 1, 1))
	netD:add(nn.VolumetricBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
	-- state size: (ndf*2) x 8 x 8 x 8
	netD:add(nn.VolumetricConvolution(ndf * 2, ndf * 4, 4, 4,4,2, 2, 2,1, 1, 1))
	netD:add(nn.VolumetricBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
	-- state size: (ndf*4) x 4 x 4 x 4
	netD:add(nn.VolumetricConvolution(ndf * 4, nz, 4, 4,4,4))
	netD:add(nn.VolumetricBatchNormalization(nz)):add(nn.LeakyReLU(0.2, true))
	-- state size: (nz)
    netD:add(nn.Reshape(nz,1,1,true))
    return netD
end

function M.VolumetricSoftMax(nC)
    -- input is B X C X H X W X D, output is also B X C X H X W X D but normalized across C
    local inpPred = -nn.Identity()
    local shift = inpPred - nn.Max(2) - nn.Unsqueeze(2) - nn.Replicate(nC,2)
    local shiftedInp = {inpPred, shift} - nn.CSubTable()
    local expInp = shiftedInp - nn.Exp()
    local denom = expInp - nn.Sum(2) - nn.Unsqueeze(2) - nn.Replicate(nC,2)
    local out = {expInp, denom} - nn.CDivTable()
    
    local gmod = nn.gModule({inpPred}, {out})
    return gmod
end

return M
