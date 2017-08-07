-- sample usage: class=chair gpu=2 th synthetic/experimentScripts.lua | bash
local params = {}
params.class = 'chair' --chair:3001627, aero:2691156, car:2958343
params.disp = 1
params.gpu = 1
params.imsave=1
for k,v in pairs(params) do params[k] = tonumber(os.getenv(k)) or os.getenv(k) or params[k] end
local classToSynset = {chair='3001627', aero='2691156', car='2958343'}
local synset = classToSynset[params.class]

-- Shapenet With Impainting Task
--local cmd = string.format('gpu=%d name=%s_impainttv2colormaskedadvheadsvoxels disp=%d synset=%s imsave=%d th synthetic/impaintMaskedTVAdvToColorShapenetVoxels.lua',params.gpu,params.class, params.disp, synset,params.imsave)
-- print(cmd)

-- ShapeNet Masked VAE Heads with Colored Voxels Experiment
--local cmd = string.format('gpu=%d name=%s_maskedvaeheadsvoxels disp=%d synset=%s imsave=%d th synthetic/maskedVaeShapenetHeadsVoxels.lua',params.gpu,params.class, params.disp, synset,params.imsave)
-- print(cmd)

-- ShapeNet TV loss Textured  Masked Conditional Adversarial Occupancy to Colored Voxels with Colored Voxels Experiment
--local cmd = string.format('gpu=%d name=%s_tvtextureocc2colormaskedcondadvheadsvoxels disp=%d synset=%s imsave=%d th synthetic/maskedTVCondAdvTextureOccToColorShapenetVoxels.lua',params.gpu,params.class, params.disp, synset,params.imsave)
-- print(cmd)


-- ShapeNet Textured  Masked Conditional Adversarial Occupancy to Colored Voxels with Colored Voxels Experiment
--local cmd = string.format('gpu=%d name=%s_textureocc2colormaskedcondadvheadsvoxels disp=%d synset=%s imsave=%d th synthetic/maskedCondAdvTextureOccToColorShapenetVoxels.lua',params.gpu,params.class, params.disp, synset,params.imsave)
-- print(cmd)

-- ShapeNet Self Supervised  Masked Conditional Adversarial Occupancy to Colored Voxels with Colored Voxels Experiment
--local cmd = string.format('gpu=%d name=%s_selfsupervisedocc2colormaskedcondadvheadsvoxels disp=%d synset=%s imsave=%d  th synthetic/maskedSelfSuperviseTextureColorShapenetVoxels.lua',params.gpu,params.class, params.disp, synset,params.imsave)
-- print(cmd)

-- ShapeNet TV Loss Masked Conditional Adversarial Occupancy to Colored Voxels with Colored Voxels Experiment
local cmd = string.format('gpu=%d name=%s_tvocc2colormaskedcondadvheadsvoxels disp=%d synset=%s imsave=%d th synthetic/maskedTVCondAdvOccToColorShapenetVoxels.lua',params.gpu,params.class, params.disp, synset,params.imsave)
-- print(cmd)

-- ShapeNet Masked Conditional Adversarial Occupancy to Colored Voxels with Colored Voxels Experiment
--local cmd = string.format('gpu=%d name=%s_occ2colormaskedcondadvheadsvoxels disp=%d synset=%s imsave=%d th synthetic/maskedCondAdvOccToColorShapenetVoxels.lua',params.gpu,params.class, params.disp, synset,params.imsave)
-- print(cmd)

-- ShapeNet TV Loss Masked Conditional Adversarial Heads with Colored Voxels Experiment
--local cmd = string.format('gpu=%d name=%s_tvmaskedcondadvheadsvoxels disp=%d synset=%s imsave=%d th synthetic/maskedTVCondAdvShapenetHeadsVoxels.lua',params.gpu,params.class, params.disp, synset,params.imsave)
-- print(cmd)

-- ShapeNet Masked Conditional Adversarial Heads with Colored Voxels Experiment
--local cmd = string.format('gpu=%d name=%s_maskedcondadvheadsvoxels disp=%d synset=%s imsave=%d th synthetic/maskedCondAdvShapenetHeadsVoxels.lua',params.gpu,params.class, params.disp, synset,params.imsave)
-- print(cmd)



-- ShapeNet  Conditional Adversarial Heads with Colored Voxels Experiment
--local cmd = string.format('gpu=%d name=%s_condadvheadsvoxels disp=%d synset=%s imsave=%d th synthetic/condAdvShapenetHeadsVoxels.lua',params.gpu,params.class, params.disp, synset,params.imsave)
-- print(cmd)



-- ShapeNet Adversarial Heads with Colored Voxels Experiment
--local cmd = string.format('gpu=%d name=%s_advheadsvoxels disp=%d synset=%s imsave=%d th synthetic/advShapenetHeadsVoxels.lua',params.gpu,params.class, params.disp, synset,params.imsave)
--print(cmd)

-- ShapeNet Masked Heads with Colored Voxels Experiment
--local cmd = string.format('gpu=%d name=%s_maskedheadsvoxels disp=%d synset=%s imsave=%d th synthetic/maskedShapenetHeadsVoxels.lua',params.gpu,params.class, params.disp, synset,params.imsave)
--print(cmd)


-- ShapeNet Heads with Colored Voxels Experiment
--local cmd = string.format('gpu=%d name=%s_headsvoxels disp=%d synset=%s imsave=%d th synthetic/shapenetHeadsVoxels.lua',params.gpu,params.class, params.disp, synset,params.imsave)
--print(cmd)


-- ShapeNet Colored Voxels Experiment
--local cmd = string.format('gpu=%d name=%s_colorvoxels disp=%d synset=%s imsave=%d th synthetic/shapenetColorVoxels.lua',params.gpu,params.class, params.disp, synset,params.imsave)
--print(cmd)


-- ShapeNet Voxels Experiment
--local cmd = string.format('gpu=%d name=%s_voxels disp=%d synset=%s imsave=%d th synthetic/shapenetVoxels.lua',params.gpu,params.class, params.disp, synset,params.imsave)
--print(cmd)
--
---- Mask Experiment
--local cmd = string.format('gpu=%d nImgs=5 maskOnly=1 name=%s_mask_nIm5 disp=%d synset=%s th synthetic/shapenet.lua',params.gpu,params.class, params.disp, synset)
--print(cmd)
--
---- Depth Experiment
--local cmd = string.format('gpu=%d nImgs=5 maskOnly=0 name=%s_depth_nIm5 disp=%d synset=%s th synthetic/shapenet.lua',params.gpu,params.class, params.disp, synset)
--print(cmd)
--
---- Noisy Depth Experiment
--local cmd = string.format('gpu=%d useNoise=1 nImgs=5 maskOnly=0 name=%s_depth_nIm5_noise disp=%d synset=%s th synthetic/shapenet.lua',params.gpu,params.class, params.disp, synset)
--print(cmd)
--
---- Fusion Experiment
--local cmd = string.format('gpu=%d name=%s_fuse_nIm5 disp=%d synset=%s th synthetic/shapenetFused.lua',params.gpu,params.class, params.disp, synset)
--print(cmd)
--
---- Fusion Experiment (Noisy Depth)
--local cmd = string.format('gpu=%d useNoise=1 name=%s_fuse_nIm5_noise disp=%d synset=%s th synthetic/shapenetFused.lua',params.gpu,params.class, params.disp, synset)
--print(cmd)
--
--
print(cmd)
