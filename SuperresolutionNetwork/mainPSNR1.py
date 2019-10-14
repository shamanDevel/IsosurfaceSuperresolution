"""
Computes the PSNR of different models
"""

from math import log10
import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np

from inference import LoadedModel
import datasetVideo
from utils import ScreenSpaceShading, initialImage
import models

from console_progressbar import ProgressBar

DATASET_PATH = '../../data/clouds/rendering_video3/'

MODEL_NEAREST = "nearest"
MODEL_BILINEAR = "bilinear"
MODEL_BICUBIC = "bicubic"
MODEL_DIR = "D:/VolumeSuperResolution"
SHADED = True
UNSHADED = False
MODELS = [
    #('Nearest-PreShade', MODEL_NEAREST, SHADED),
    #('Bilinear-PreShade', MODEL_BILINEAR, SHADED),
    #('Bicubic-PreShade', MODEL_BICUBIC, SHADED),
    ('Nearest', MODEL_NEAREST, UNSHADED),
    ('Bilinear', MODEL_BILINEAR, UNSHADED),
    ('Bicubic', MODEL_BICUBIC, UNSHADED),
    ('GAN 1', 'pretrained_unshaded/gen_gan_1.pth', UNSHADED),
    ('GAN 2', 'pretrained_unshaded/gen_gan_2b_nAO_wCol.pth', UNSHADED),
    ('L1-normal', 'pretrained_unshaded/gen_l1normal.pth', UNSHADED),
    #('Shaded-GAN', 'pretrained_unshaded/shadedGenerator_EnhanceNet_percp+tl2+tgan1.pth', SHADED)
    ]

device = torch.device("cuda")

# Load dataset
opt_dict = dict()
opt_dict['unshaded'] = True
opt_dict['aoInverted'] = False
opt_dict['upscale_factor'] = 4
opt_dict['inputPathUnshaded'] = DATASET_PATH
opt_dict['inputPathShaded'] = None
opt_dict['numberOfImages'] = 5
opt_dict['samples'] = 5000
dataset_data = datasetVideo.collect_samples_clouds_video(
        opt_dict['upscale_factor'], opt_dict, deferred_shading=True)

class DatasetFromFullImagesAll(data.Dataset):
    def __init__(self, 
                 dataset_data, #The images
                 num_images):
        super(DatasetFromFullImagesAll, self).__init__()

        self.data = dataset_data
        self.num_images = num_images

    def __getitem__(self, index):
        return torch.from_numpy(self.data.images_low[index]), \
               torch.from_numpy(self.data.flow_low[index]),   \
               torch.from_numpy(self.data.images_high[index])

    def __len__(self):
        return self.num_images

#test_full_set = DatasetFromFullImagesAll(dataset_data, len(dataset_data.images_low))
#data_loader = data.DataLoader(dataset=test_full_set, batch_size=1, shuffle=False)
test_set = datasetVideo.DatasetFromSamples(dataset_data, True, 0.2)
data_loader = data.DataLoader(dataset=test_set, batch_size=16, shuffle=False)

# Load models
class SimpleUpsample(nn.Module):
    def __init__(self, upscale_factor, upsample, shaded):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.input_channels = 3 if shaded else 5
        self.channel_mask = [0,1,2] if shaded else [0,1,2,3,4,5]
        self.output_channels = 3 if shaded else 6
        self.upsample = upsample
        self.shaded = shaded

    def forward(self, inputs):
        channel_mask_length = len(self.channel_mask)
        inputs_masked = inputs[:,0:channel_mask_length,:,:] #time per hit: 289.8
        resized_inputs = F.interpolate(inputs_masked, 
                                       size=[inputs.shape[2]*self.upscale_factor, 
                                             inputs.shape[3]*self.upscale_factor], 
                                       mode=self.upsample)
        if channel_mask_length==self.output_channels:
            return resized_inputs, None
        elif channel_mask_length<self.output_channels:
            return torch.cat([
                resized_inputs,
                torch.zeros(resized_inputs.shape[0],
                            channel_mask_length-self.output_channels,
                            resized_inputs.shape[2],
                            resized_inputs.shape[3],
                            dtype=resized_inputs.dtype,
                            device=resized_inputs.device)],
                dim=1), None
        else:
            raise ValueError("number of output channels must be at least the number of masked input channels")

print('Load Models')
modelList = [None] * len(MODELS)
for i, (name, p, shaded) in enumerate(MODELS):
    if p==MODEL_NEAREST or p==MODEL_BILINEAR or p==MODEL_BICUBIC:
        mode = p
        modelList[i] = SimpleUpsample(4, mode, shaded)
        modelList[i].to(device)
    else:
        modelList[i] = LoadedModel(os.path.join(MODEL_DIR, p), device, 4).model

# create shading
shading = ScreenSpaceShading(device)
shading.fov(30)
shading.ambient_light_color(np.array([0.1,0.1,0.1]))
shading.diffuse_light_color(np.array([1.0, 1.0, 1.0]))
shading.specular_light_color(np.array([0.2, 0.2, 0.2]))
shading.specular_exponent(16)
shading.light_direction(np.array([0.1,0.1,1.0]))
shading.material_color(np.array([1.0, 0.3, 0.0]))
shading.ambient_occlusion(1.0)
shading.inverse_ao = False

# perform measurements
psnr = [0.0] * len(MODELS)
for model in range(len(MODELS)):
    print('Measure', MODELS[model][0])
    with torch.no_grad():
        num_minibatch = len(data_loader)
        pg = ProgressBar(num_minibatch, 'Test %d Images'%num_minibatch, length=50)
        for i,batch in enumerate(data_loader):
            pg.print_progress_bar(i)
            input, flow, high = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            B, _, Cin, H, W = input.shape
            Hhigh = H * 4
            Whigh = W * 4
            previous_output = None
            # loop over frames
            for j in range(dataset_data.num_frames):
                # prepare input
                if j == 0:
                    previous_warped = initialImage(input[:,0,:,:,:], 6, 
                                               'zero', False, 4)
                else:
                    previous_warped = models.VideoTools.warp_upscale(
                        previous_output, 
                        flow[:, j-1, :, :, :], 
                        4,
                        special_mask = True)
                previous_warped_flattened = models.VideoTools.flatten_high(previous_warped, 4)
                single_input = torch.cat((
                        input[:,j,:,:,:],
                        previous_warped_flattened),
                    dim=1)
                # run model
                prediction, _ = modelList[model](single_input)
                # shade output
                prediction[:,1:4,:,:] = ScreenSpaceShading.normalize(prediction[:,1:4,:,:], dim=1)
                shaded_prediction = shading(prediction)
                # shade high res
                shaded_gt = shading(high[:,j,:,:,:])
                # compute psnr
                mse = F.mse_loss(shaded_prediction, shaded_gt).item()
                psnr[model] += 10 * log10(1 / mse)
                # save output for next frame
                previous_output = torch.cat([
                    torch.clamp(prediction[:,0:1,:,:], -1, +1), # mask
                    ScreenSpaceShading.normalize(prediction[:,1:4,:,:], dim=1),
                    torch.clamp(prediction[:,4:5,:,:], 0, +1), # depth
                    torch.clamp(prediction[:,5:6,:,:], 0, +1) # ao
                    ], dim=1)
        pg.print_progress_bar(num_minibatch)
        psnr[model] /= (num_minibatch * dataset_data.num_frames)

# print PSNR
print("Model & PSNR")
for i, (name, path, shaded) in enumerate(MODELS):
    print("%s & %5.3f"%(name,psnr[i]))
