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

OUTPUT_FOLDER = "../results/"

DATASET_PREFIX = "../../data/"
DATASETS = [
    ("Clouds", ["clouds/rendering_video3"]),
    #("Thorax", ["volumes/rendering/cleveland70"]),
    ("RM", ["volumes/rendering/ppmt145_128_border", 
            "volumes/rendering/ppmt145_256_border", 
            "volumes/rendering/ppmt145_512_border", 
            "volumes/rendering/ppmt273_512_border",
            "volumes/rendering/ppmt273_1024_border"]),
    ("Ejecta", ["volumes/rendering/snapshot_0%d0_256"%i for i in range(1, 10)] + \
               ["volumes/rendering/snampshot_272_512_ushort"]),
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
    #{
    #    'name': 'gen_l1normal',
    #    'path': 'pretrained_unshaded/gen_l1normal.pth',
    #    'temporal': True,
    #    'masking': True
    #},
    {
        'name': 'gen_l1normal_clouds',
        'path': 'pretrained_unshaded/gen_l1normalDepth_2.pth',
        'temporal': True,
        'masking': True
    },
    {
        'name': 'gen_l1normal_ejecta',
        'path': 'pretrained_unshaded/gen_l1normal_allEjecta_epoch_100.pth',
        'temporal': True,
        'masking': True
    },
    ]
UPSCALING = 4

# Load models
device = torch.device("cuda")
class SimpleUpsample(nn.Module):
    def __init__(self, upscale_factor, upsample):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.input_channels = 5
        self.output_channels = 6
        self.upsample = upsample

    def forward(self, inputs):
        inputs = inputs[:,0:self.input_channels,:,:]
        resized_inputs = F.interpolate(inputs, 
                                       size=[inputs.shape[2]*self.upscale_factor, 
                                             inputs.shape[3]*self.upscale_factor], 
                                       mode=self.upsample)

        return torch.cat([
            resized_inputs,
            torch.ones(resized_inputs.shape[0],
                        self.output_channels - self.input_channels,
                        resized_inputs.shape[2],
                        resized_inputs.shape[3],
                        dtype=resized_inputs.dtype,
                        device=resized_inputs.device)],
            dim=1), None
model_list = [None]*len(MODELS)
for i,m in enumerate(MODELS):
    p = m['path']
    if p is not None:
        model_list[i] = inference.LoadedModel(os.path.join(MODEL_DIR,p), device, UPSCALING).model
    else:
        model_list[i] = SimpleUpsample(UPSCALING, m["name"])

# create output folder
os.makedirs(OUTPUT_FOLDER, exist_ok = True)

# create shading
shading = ScreenSpaceShading(device)
shading.fov(30)
shading.ambient_light_color(np.array([0.1,0.1,0.1]))
shading.diffuse_light_color(np.array([1.0, 1.0, 1.0]))
shading.specular_light_color(np.array([0.0, 0.0, 0.0]))
shading.specular_exponent(16)
shading.light_direction(np.array([0.1,0.1,1.0]))
shading.material_color(np.array([1.0, 0.3, 0.0]))
AMBIENT_OCCLUSION_STRENGTH = 1.0
shading.ambient_occlusion(1.0)
shading.inverse_ao = False

# STATISTICS
ssimLoss = MSSSIM()
ssimLoss.to(device)
psnrLoss = PSNR()
psnrLoss.to(device)
BORDER = 15
MIN_FILLING = 0.05
NUM_BINS = 200
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
        self.histogram_depth = np.zeros(NUM_BINS, dtype=np.float64)
        self.histogram_normal = np.zeros(NUM_BINS, dtype=np.float64)
        self.histogram_mask = np.zeros(NUM_BINS, dtype=np.float64)
        self.histogram_ao = np.zeros(NUM_BINS, dtype=np.float64)
        self.histogram_counter = 0

    def reset(self):
        self.n = 0

        self.psnr_normal = 0
        self.psnr_depth = 0
        self.psnr_ao = 0
        self.psnr_color_withAO = 0
        self.psnr_color_noAO = 0

        self.ssim_normal = 0
        self.ssim_depth = 0
        self.ssim_ao = 0
        self.ssim_color_withAO = 0
        self.ssim_color_noAO = 0

        self.l2ds_normal_mean = 0
        self.l2ds_normal_max = 0
        self.l2ds_colorNoAO_mean = 0
        self.l2ds_colorNoAO_max = 0

    def write_header(self, file):
        file.write("PSNR-normal\tPSNR-depth\tPSNR-ao\tPSNR-color-noAO\tPSNR-color-withAO\t")
        file.write("SSIM-normal\tSSIM-depth\tSSIM-ao\tSSIM-color-noAO\tSSIM-color-withAO\t")
        file.write("L2-ds-normal-mean\tL2-ds-normal-max\tL2-ds-color-noAO-mean\tL2-ds-color-noAO-max\n")

    def add_timestep_sample(self, pred_mnda, gt_mnda, input_mnda):
        """
        adds a timestep sample:
        pred_mnda: prediction: mask, normal, depth, AO
        pred_color: shaded color
        gt_mnda: ground truth: mask, normal, depth, AO
        gt_color: shaded ground truth
        """

        #shading
        shading.ambient_occlusion(AMBIENT_OCCLUSION_STRENGTH)
        pred_color_withAO = shading(pred_mnda)
        gt_color_withAO = shading(gt_mnda)
        shading.ambient_occlusion(0.0)
        pred_color_noAO = shading(pred_mnda)
        gt_color_noAO = shading(gt_mnda)
        input_color_noAO = shading(input_mnda)

        #apply border
        BORDER2 = BORDER * UPSCALING
        pred_mnda = pred_mnda[:,:,BORDER2:-BORDER2,BORDER2:-BORDER2]
        pred_color_withAO = pred_color_withAO[:,:,BORDER2:-BORDER2,BORDER2:-BORDER2]
        pred_color_noAO = pred_color_noAO[:,:,BORDER2:-BORDER2,BORDER2:-BORDER2]
        gt_mnda = gt_mnda[:,:,BORDER2:-BORDER2,BORDER2:-BORDER2]
        gt_color_withAO = gt_color_withAO[:,:,BORDER2:-BORDER2,BORDER2:-BORDER2]
        gt_color_noAO = gt_color_noAO[:,:,BORDER2:-BORDER2,BORDER2:-BORDER2]
        input_mnda = input_mnda[:,:,BORDER:-BORDER,BORDER:-BORDER]
        input_color_noAO = input_color_noAO[:,:,BORDER:-BORDER,BORDER:-BORDER]
        
        #self.psnr_normal += 10 * math.log10(1 / torch.mean((pred_mnda[0,1:4,:,:]-gt_mnda[0,1:4,:,:])**2).item())
        #self.psnr_ao += 10 * math.log10(1 / torch.mean((pred_mnda[0,5:6,:,:]-gt_mnda[0,5:6,:,:])**2).item())
        #self.psnr_color += 10 * math.log10(1 / torch.mean((pred_color-gt_color)**2).item())

        # check fill rate
        mask = gt_mnda[:,0:1,:,:] * 0.5 + 0.5
        B,C,H,W = mask.shape
        factor = torch.sum(mask).item() / (H*W)
        if factor < MIN_FILLING:
            #print("Ignore, too few filled points")
            return

        self.n += 1

        # PSNR
        self.psnr_normal += psnrLoss(pred_mnda[:,1:4,:,:], gt_mnda[:,1:4,:,:], mask=mask).item()
        self.psnr_depth += psnrLoss(pred_mnda[:,4:5,:,:], gt_mnda[:,4:5,:,:], mask=mask).item()
        self.psnr_ao += psnrLoss(pred_mnda[:,5:6,:,:], gt_mnda[:,5:6,:,:], mask=mask).item()
        self.psnr_color_withAO += psnrLoss(pred_color_withAO, gt_color_withAO, mask=mask).item()
        self.psnr_color_noAO += psnrLoss(pred_color_noAO, gt_color_noAO, mask=mask).item()

        # SSIM
        pred_mnda = gt_mnda + mask * (pred_mnda - gt_mnda)
        self.ssim_normal += ssimLoss(pred_mnda[:,1:4,:,:], gt_mnda[:,1:4,:,:]).item()
        self.ssim_depth += ssimLoss(pred_mnda[:,4:5,:,:], gt_mnda[:,4:5,:,:]).item()
        self.ssim_ao += ssimLoss(pred_mnda[:,5:6,:,:], gt_mnda[:,5:6,:,:]).item()
        self.ssim_color_withAO += ssimLoss(pred_color_withAO, gt_color_withAO).item()
        self.ssim_color_noAO += ssimLoss(pred_color_noAO, gt_color_noAO).item()

        # Downsample Loss
        ds_normal = self.downsample_loss(
            input_mnda[:,1:4,:,:], 
            ScreenSpaceShading.normalize(self.downsample(pred_mnda[:,1:4,:,:]), dim=1))
        ds_color = self.downsample_loss(
            input_color_noAO, 
            self.downsample(pred_color_noAO))
        self.l2ds_normal_mean += torch.mean(ds_normal).item()
        self.l2ds_normal_max = max(self.l2ds_normal_max, torch.max(ds_normal).item())
        self.l2ds_colorNoAO_mean += torch.mean(ds_color).item()
        self.l2ds_colorNoAO_max = max(self.l2ds_colorNoAO_max, torch.max(ds_color).item())

        # Histogram
        self.histogram_counter += 1

        mask_diff = F.l1_loss(gt_mnda[0,0,:,:], pred_mnda[0,0,:,:], reduction='none')
        histogram,_ = np.histogram(mask_diff.cpu().numpy(), bins=NUM_BINS, range=(0,1), density=True)
        self.histogram_mask += (histogram/NUM_BINS - self.histogram_mask)/self.histogram_counter

        #normal_diff = (-F.cosine_similarity(gt_mnda[0,1:4,:,:], pred_mnda[0,1:4,:,:], dim=0)+1)/2
        normal_diff = F.l1_loss(gt_mnda[0,1:4,:,:], pred_mnda[0,1:4,:,:], reduction='none').sum(dim=0) / 6
        histogram,_ = np.histogram(normal_diff.cpu().numpy(), bins=NUM_BINS, range=(0,1), density=True)
        self.histogram_normal += (histogram/NUM_BINS - self.histogram_normal)/self.histogram_counter

        depth_diff = F.l1_loss(gt_mnda[0,4,:,:], pred_mnda[0,4,:,:], reduction='none')
        histogram,_ = np.histogram(depth_diff.cpu().numpy(), bins=NUM_BINS, range=(0,1), density=True)
        self.histogram_depth += (histogram/NUM_BINS - self.histogram_depth)/self.histogram_counter

        ao_diff = F.l1_loss(gt_mnda[0,5,:,:], pred_mnda[0,5,:,:], reduction='none')
        histogram,_ = np.histogram(ao_diff.cpu().numpy(), bins=NUM_BINS, range=(0,1), density=True)
        self.histogram_ao += (histogram/NUM_BINS - self.histogram_ao)/self.histogram_counter

        color_diff = F.l1_loss(gt_color_withAO[0,0,:,:], pred_color_withAO[0,0,:,:], reduction='none')
        histogram,_ = np.histogram(color_diff.cpu().numpy(), bins=NUM_BINS, range=(0,1), density=True)
        self.histogram_color_withAO += (histogram/NUM_BINS - self.histogram_color_withAO)/self.histogram_counter

        color_diff = F.l1_loss(gt_color_noAO[0,0,:,:], pred_color_noAO[0,0,:,:], reduction='none')
        histogram,_ = np.histogram(color_diff.cpu().numpy(), bins=NUM_BINS, range=(0,1), density=True)
        self.histogram_color_noAO += (histogram/NUM_BINS - self.histogram_color_noAO)/self.histogram_counter

    def write_sample(self, file):
        """
        All timesteps were added, write out the statistics for this sample
        and reset.
        """
        self.n = max(1, self.n)
        file.write("%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%e\t%e\t%e\t%e\n" % (
            self.psnr_normal/self.n, self.psnr_depth/self.n, self.psnr_ao/self.n, self.psnr_color_noAO/self.n, self.psnr_color_withAO/self.n,
            self.ssim_normal/self.n, self.ssim_depth/self.n, self.ssim_ao/self.n, self.ssim_color_noAO/self.n, self.ssim_color_withAO/self.n,
            self.l2ds_normal_mean/self.n, self.l2ds_normal_max, self.l2ds_colorNoAO_mean/self.n, self.l2ds_colorNoAO_max))
        file.flush()
        self.reset()

    def write_histogram(self, file):
        """
        After every sample for the current dataset was processed, write
        a histogram of the errors in a new file
        """
        file.write("BinStart\tBinEnd\tL2ErrorMask\tCosineErrorNormal\tL2ErrorDepth\tL2ErrorAO\tL2ErrorColorWithAO\tL2ErrorColorNoAO\n")
        for i in range(NUM_BINS):
            file.write("%7.5f\t%7.5f\t%e\t%e\t%e\t%e\t%e\t%e\n" % (
                i / NUM_BINS, (i+1) / NUM_BINS,
                self.histogram_mask[i],
                self.histogram_normal[i],
                self.histogram_depth[i],
                self.histogram_ao[i],
                self.histogram_color_withAO[i],
                self.histogram_color_noAO[i]
                ))
        pass

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
                            previous_warped = initialImage(sample_low[0:1,:,:,:], 6, 'zero', False, UPSCALING)
                        else:
                            previous_warped = models.VideoTools.warp_upscale(
                                previous_output, 
                                sample_flow[j-1:j, :, :, :], 
                                UPSCALING,
                                special_mask = True)
                            #previous_warped = previous_output
                        previous_warped_flattened = models.VideoTools.flatten_high(previous_warped, UPSCALING)
                        single_input = torch.cat((
                                sample_low[j:j+1,:,:,:],
                                previous_warped_flattened),
                            dim=1)
                        # run model
                        prediction, _ = model_list[model_index](single_input)
                        prediction[:,0:1,:,:] = torch.clamp(prediction[:,0:1,:,:], -1, +1) #mask
                        prediction[:,1:4,:,:] = ScreenSpaceShading.normalize(prediction[:,1:4,:,:], dim=1) # normal
                        prediction[:,4:6,:,:] = torch.clamp(prediction[:,4:6,:,:], 0, +1) # depth+ao

                        ##DEBUG: save the images
                        #img_rgb = (prediction[:,1:4,:,:] * 0.5 + 0.5)[0].cpu().numpy()
                        ##img_rgb = img_rgb.transpose((2,1,0))
                        #scipy.misc.toimage(img_rgb, cmin=0.0, cmax=1.0).save(
                        #    os.path.join(OUTPUT_FOLDER, "Img_%s_%s_%d.png"%(dataset_name, MODELS[model_index]['name'], j)))
                        #if model_index==0:
                        #    img_rgb = (sample_high[j,1:4,:,:] * 0.5 + 0.5).cpu().numpy()
                        #    scipy.misc.toimage(img_rgb, cmin=0.0, cmax=1.0).save(
                        #        os.path.join(OUTPUT_FOLDER, "Img_%s_GT_%d.png"%(dataset_name,j)))


                        # STATISTICS
                        statistics[model_index].add_timestep_sample(
                            prediction, sample_high[j:j+1,:,:,:], sample_low[j:j+1,:,:,:])

                        # POST
                        # save output for next frame
                        previous_output = prediction
                        #break

                    # write statistics
                    statistics[model_index].write_sample(stats_models[model_index])

                # log progress
                image_index += 1
                if image_index%10 == 0:
                    print(" %d"%image_index, end='', flush=True)
    
    for k,m in enumerate(MODELS):
        with open(os.path.join(OUTPUT_FOLDER, "Histogram_%s_%s.txt"%(dataset_name, m['name'])), "w") as histo_file:
            statistics[k].write_histogram(histo_file)
    print()
    for file in stats_models:
        file.close()
