
"""
Computes various statistics for different datasets
"""

import math
import os
import os.path
import time

import numpy as np
import scipy.misc
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F

import imageio
from PIL import ImageFont, ImageDraw, Image
from console_progressbar import ProgressBar
from matplotlib.pyplot import imshow

import inference, models
from losses import LossBuilder
from utils import ScreenSpaceShading, initialImage, MeanVariance, MSSSIM, PSNR

OUTPUT_FOLDER = "../results2/"

DATASET_PREFIX = "../../data/"
DATASETS = [
    ("Clouds", ["clouds/rendering_video3"]),
    #("Thorax", ["volumes/rendering/cleveland70"]),
    #("RM", ["volumes/rendering/ppmt145_128_border", 
    #        "volumes/rendering/ppmt145_256_border", 
    #        "volumes/rendering/ppmt145_512_border", 
    #        "volumes/rendering/ppmt273_512_border",
    #        "volumes/rendering/ppmt273_1024_border"]),
    #("Ejecta", ["volumes/rendering/snapshot_0%d0_256"%i for i in range(1, 10)] + \
    #           ["volumes/rendering/snampshot_272_512_ushort"]),
    #("Skull", ["volumes/rendering/vmhead256cubed"])
    ]

MODEL_NEAREST = "nearest"
MODEL_BILINEAR = "bilinear"
MODEL_BICUBIC = "bicubic"
MODEL_DIR = "D:/VolumeSuperResolution"
MODELS = [
    {'name' : MODEL_NEAREST, 'path' : None},
    {'name' : MODEL_BILINEAR, 'path' : None},
    {'name' : MODEL_BICUBIC, 'path' : None},
    
    {
        'name': 'shadedOutput',
        'path': 'pretrained_unshaded/shadedGenerator_EnhanceNet_percp+tl2+tgan1.pth',
        'temporal': True,
        'masking': True
    },
    {
        'name': 'gen_l1color',
        'path': 'pretrained_unshaded/gen_l1color.pth',
        'temporal': True,
        'masking': True
    },

    {
        'name': 'gen_l1normalDepth_clouds',
        'path': 'pretrained_unshaded/gen_l1normalDepth_2.pth',
        'temporal': True,
        'masking': True
    },
    {
        'name': 'gen_l1normalDepth_ejecta',
        'path': 'pretrained_unshaded/gen_l1normal_allEjecta_epoch_100.pth',
        'temporal': True,
        'masking': True
    },
    {
        'name': 'gen_percNormalDepth',
        'path': 'pretrained_unshaded/gen_percDepthNormal.pth',
        'temporal': True,
        'masking': True
    },
    {
        'name': 'gen_gan3',
        'path': 'pretrained_unshaded/gen_gan3.pth',
        'temporal': True,
        'masking': True
    },
    {
        'name': 'gen_l1normalDepth_rm',
        'path': 'pretrained_unshaded/gen_l1normalDepth_allRM.pth',
        'temporal': True,
        'masking': True
    },
    ]
UPSCALING = 4
device = torch.device("cuda")

# create shading
shading = ScreenSpaceShading(device)
shading.fov(30)
shading.ambient_light_color(np.array([0.1,0.1,0.1]))
shading.diffuse_light_color(np.array([0.9, 0.9, 0.9]))
shading.specular_light_color(np.array([0.02, 0.02, 0.02]))
shading.specular_exponent(16)
shading.light_direction(np.array([0.1,0.1,1.0]))
shading.material_color(np.array([1.0, 1.0, 1.0]))
shading.ambient_occlusion(0.0)
shading.inverse_ao = False

# Load models
class SimpleUpsample(nn.Module):
    def __init__(self, upscale_factor, upsample, shading):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.input_channels = 5
        self.output_channels = 6
        self.upsample = upsample
        self.shading = shading
        self.prev_input_channels = 6

    def forward(self, sample_low, previous_warped_flattened):
        inputs = sample_low[:,0:self.input_channels,:,:]
        resized_inputs = F.interpolate(inputs, 
                                       size=[inputs.shape[2]*self.upscale_factor, 
                                             inputs.shape[3]*self.upscale_factor], 
                                       mode=self.upsample)

        prediction = torch.cat([
            resized_inputs,
            torch.ones(resized_inputs.shape[0],
                        self.output_channels - self.input_channels,
                        resized_inputs.shape[2],
                        resized_inputs.shape[3],
                        dtype=resized_inputs.dtype,
                        device=resized_inputs.device)],
            dim=1)
        prediction[:,0:1,:,:] = torch.clamp(prediction[:,0:1,:,:], -1, +1) #mask
        prediction[:,1:4,:,:] = ScreenSpaceShading.normalize(prediction[:,1:4,:,:], dim=1) # normal
        prediction[:,4:6,:,:] = torch.clamp(prediction[:,4:6,:,:], 0, +1) # depth+ao

        color = self.shading(prediction)
        return color, prediction

class ShadedModel(nn.Module):
    def __init__(self, upscale_factor, mdl, shading):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.mdl = mdl
        self.shading = shading
        self.prev_input_channels = 3

    def forward(self, sample_low, previous_warped_flattened):
        #sample low contains mask+normal+depth
        # but we need color+mask(in 0,1)+normal+depth
        sample_low = torch.cat([
            self.shading(sample_low),
            sample_low[:,0:1,:,:]*0.5+0.5, #transform mask back to [0,1]
            sample_low[:,1:4,:,:],
            sample_low[:,4:5,:,:]], dim=1) 
        single_input = torch.cat((
                sample_low,
                previous_warped_flattened),
            dim=1)

        color,_ = self.mdl(single_input)
        return torch.clamp(color, 0, 1), torch.clamp(color, 0, 1)

class UnshadedModel(nn.Module):
    def __init__(self, upscale_factor, mdl, shading):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.mdl = mdl
        self.shading = shading
        self.prev_input_channels = 6

    def forward(self, sample_low, previous_warped_flattened):
        single_input = torch.cat((
                sample_low,
                previous_warped_flattened),
            dim=1)

        prediction,_ = self.mdl(single_input)

        prediction[:,0:1,:,:] = torch.clamp(prediction[:,0:1,:,:], -1, +1) #mask
        prediction[:,1:4,:,:] = ScreenSpaceShading.normalize(prediction[:,1:4,:,:], dim=1) # normal
        prediction[:,4:6,:,:] = torch.clamp(prediction[:,4:6,:,:], 0, +1) # depth+ao

        color = self.shading(prediction)
        return color, prediction

model_list = [None]*len(MODELS)
for i,m in enumerate(MODELS):
    p = m['path']
    if p is not None:
        mdl = inference.LoadedModel(os.path.join(MODEL_DIR,p), device, UPSCALING)
        if mdl.unshaded:
            model_list[i] = UnshadedModel(UPSCALING, mdl.model, shading)
        else:
            model_list[i] = ShadedModel(UPSCALING, mdl.model, shading)
    else:
        model_list[i] = SimpleUpsample(UPSCALING, m["name"], shading)

# create output folder
os.makedirs(OUTPUT_FOLDER, exist_ok = True)

# STATISTICS
ssimLoss = MSSSIM()
ssimLoss.to(device)
psnrLoss = PSNR()
psnrLoss.to(device)
BORDER = 15
MIN_FILLING = 0.05
NUM_BINS = 100
class Statistics:

    def __init__(self):
        self.reset()
        self.downsample = nn.Upsample(
            scale_factor = 1.0/UPSCALING, mode='bilinear')
        self.downsample_loss = nn.MSELoss(reduction='none')
        self.downsample.to(device)
        self.downsample_loss.to(device)

        self.histogram_color_withAO = np.zeros(NUM_BINS, dtype=np.float64)
        self.histogram_color_noAO = np.zeros(NUM_BINS, dtype=np.float64)
        self.histogram_counter = 0

    def reset(self):
        self.n = 0
        self.psnr_color = 0
        self.ssim_color = 0

    def write_header(self, file):
        file.write("PSNR-color\tSSIM-color\n")

    def add_timestep_sample(self, pred_color, gt_mnda):
        """
        adds a timestep sample:
        pred_mnda: prediction: mask, normal, depth, AO
        pred_color: shaded color
        gt_mnda: ground truth: mask, normal, depth, AO
        gt_color: shaded ground truth
        """

        #shading
        gt_color = shading(gt_mnda)

        #apply border
        BORDER2 = BORDER * UPSCALING
        gt_mnda = gt_mnda[:,:,BORDER2:-BORDER2,BORDER2:-BORDER2]
        pred_color = pred_color[:,:,BORDER2:-BORDER2,BORDER2:-BORDER2]
        gt_color = gt_color[:,:,BORDER2:-BORDER2,BORDER2:-BORDER2]

        # check fill rate
        mask = gt_mnda[:,0:1,:,:] * 0.5 + 0.5
        B,C,H,W = mask.shape
        factor = torch.sum(mask).item() / (H*W)
        if factor < MIN_FILLING:
            #print("Ignore, too few filled points")
            return

        self.n += 1

        # PSNR
        self.psnr_color += psnrLoss(pred_color, gt_color, mask=mask).item()

        # SSIM
        self.ssim_color += ssimLoss(pred_color, gt_color).item()

    def write_sample(self, file):
        """
        All timesteps were added, write out the statistics for this sample
        and reset.
        """
        self.n = max(1, self.n)
        file.write("%.6f\t%.6f\n" % (
            self.psnr_color/self.n, self.ssim_color/self.n))
        file.flush()
        self.reset()


# Compute statistics for every dataset
for dataset_name, dataset_folders in DATASETS:
    print("Compute statistics for", dataset_name)
    stats_models = [
        open(os.path.join(OUTPUT_FOLDER, "Stats_%s_%s.txt"%(dataset_name, m['name'])), "w") \
        for m in MODELS]

    statistics = [Statistics() for i in range(len(MODELS))]

    for k,file in enumerate(stats_models):
        statistics[k].write_header(file)

    image_index = 0
    with torch.no_grad():
        for dataset_folder in dataset_folders:
            # iterate over all samples
            folder = os.path.join(DATASET_PREFIX, dataset_folder)
            for i in range(10000):
                file_low = os.path.join(folder, "low_%05d.npy"%i)
                if not os.path.isfile(file_low):
                    break
                file_high = os.path.join(folder, "high_%05d.npy"%i)
                file_flow = os.path.join(folder, "flow_%05d.npy"%i)
                # load them
                sample_low = torch.from_numpy(np.load(file_low)).to(device)
                sample_high = torch.from_numpy(np.load(file_high)).to(device)
                sample_flow = torch.from_numpy(np.load(file_flow)).to(device)
                # iterate over all models
                for model_index in range(len(MODELS)):
                    statistics[model_index].reset();
                    # iterate over time
                    NF, C, H, W = sample_low.shape
                    for j in range(NF):
                        # SUPER-RES
                        # prepare input
                        if j == 0:
                            previous_warped = initialImage(
                                sample_low[0:1,:,:,:], 
                                model_list[model_index].prev_input_channels, 
                                'zero', False, UPSCALING)
                        else:
                            previous_warped = models.VideoTools.warp_upscale(
                                previous_output, 
                                sample_flow[j-1:j, :, :, :], 
                                UPSCALING,
                                special_mask = True)
                        previous_warped_flattened = models.VideoTools.flatten_high(previous_warped, UPSCALING)
                        # run model
                        pred_color, previous_output = model_list[model_index](sample_low[j:j+1,:,:,:], previous_warped_flattened)
                        
                        ##DEBUG: save the images
                        #img_rgb = pred_color[0].cpu().numpy()
                        ##img_rgb = img_rgb.transpose((2,1,0))
                        #scipy.misc.toimage(img_rgb, cmin=0.0, cmax=1.0).save(
                        #    os.path.join(OUTPUT_FOLDER, "Img_%s_%s_%d.png"%(dataset_name, MODELS[model_index]['name'], j)))
                        #if model_index==0:
                        #    img_rgb = (shading(sample_high[j:j+1,:,:,:])[0]).cpu().numpy()
                        #    scipy.misc.toimage(img_rgb, cmin=0.0, cmax=1.0).save(
                        #        os.path.join(OUTPUT_FOLDER, "Img_%s_GT_%d.png"%(dataset_name,j)))


                        # STATISTICS
                        statistics[model_index].add_timestep_sample(
                            pred_color, sample_high[j:j+1,:,:,:])
                        #break

                    # write statistics
                    statistics[model_index].write_sample(stats_models[model_index])

                # log progress
                image_index += 1
                if image_index%10 == 0:
                    print(" %d"%image_index, end='', flush=True)
    
    print()
    for file in stats_models:
        file.close()
