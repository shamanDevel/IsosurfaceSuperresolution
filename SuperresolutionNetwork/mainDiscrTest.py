"""
Test file that prints the discriminator response for different data
"""

import math
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import datasetVideo
import models
import losses

# './modeldir_video/run00110/model_epoch_1.pth' : WT-GAN
# './modeldir_video/run00111/model_epoch_1.pth' : T-GAN
# './modeldir_video/run00112/model_epoch_1.pth' : GAN
# './modeldir_video/run00113/model_epoch_1.pth' : W-GAN

opt = {
    'dataset':'cloud-video',
    'upscale_factor':4,
    'numberOfImages':1,
    'samples':4,
    'useInputDepth':True,
    'useInputNormal':True,
    'generatorPath':'./pretrained/generator_TecoGAN_l2_bilinear.pth',
    'discriminatorPath':'pretrained/discriminator_EnhanceNetLarge_wgan-gp.pth',
    'discriminator':'enhanceNetLarge',
    'discriminatorUsePreviousImage':False,
    'lossBorderPadding':16,
    'cuda':True
    }
opt = argparse.Namespace(**opt)

# Load data and models
device = torch.device('cuda')
dataset_data = datasetVideo.collect_samples_clouds_video(opt.upscale_factor, opt)
train_set = datasetVideo.DatasetFromSamples(dataset_data, False, 0.5)
gen = torch.load(opt.generatorPath)['model']
gen.to(device)
discr, loss = losses.LossBuilder(device).wgan_loss(
    opt.discriminator, 
    train_set.get_high_res_shape()[1],
    dataset_data.input_channels + (2 if opt.discriminatorUsePreviousImage else 1)*dataset_data.output_channels)
if opt.discriminatorPath is not None:
    discr.load_state_dict(torch.load(opt.discriminatorPath)['discriminator'])
discr.to(device)
loss.to(device)

# evaluate discriminator
input, flow, target = train_set[0]
input = torch.unsqueeze(input, 0).to(device)
flow = torch.unsqueeze(flow, 0).to(device)
target = torch.unsqueeze(target, 0).to(device)
B, _, Cout, Hhigh, Whigh = target.shape
_, _, Cin, H, W = input.shape

def evalDiscr(img, j, txt):
    previous_warped_loss = torch.zeros(B, Cout, Hhigh, Whigh, 
                            dtype=input.dtype,
                            device=device)
    input_high = F.interpolate(input[:,j,:,:,:], 
                               size=(target.shape[-2],target.shape[-1]),
                               mode='bilinear')
    if opt.discriminatorUsePreviousImage:
        input_images = torch.cat([
            torch.cat([target[:,j,:,:,:], input_high, previous_warped_loss], dim=1), 
            torch.cat([img, input_high, previous_warped_loss], dim=1)
            ], dim=0)
    else:
        input_images = torch.cat([
            torch.cat([target[:,j,:,:,:], input_high], dim=1), 
            torch.cat([img, input_high], dim=1)
            ], dim=0)
    input_images = losses.LossNet.pad(input_images, opt.lossBorderPadding)
    logit = discr(input_images)
    adv_g_loss, adv_d_loss, mean_gt_logits, mean_pred_logits = loss(logit)
    print()
    print(txt)
    print('  adv_g_loss:', adv_g_loss.item())
    print('  adv_d_loss:', adv_d_loss.item())
    print('  mean_gt_logits:', mean_gt_logits)
    print('  mean_pred_logits:', mean_pred_logits)

with torch.no_grad():
    j = 1
    evalDiscr(F.interpolate(input[:,j,:,:,:], 
                            size=(target.shape[-2],target.shape[-1]),
                            mode='bicubic')[:,0:3,:,:],
              1, 'Input:')

    evalDiscr(target[:,j,:,:,:], 1, 'Ground Truth:')

    single_input = torch.cat((
            input[:,j,:,:,:],
            torch.zeros(B, Cout*16, H, W, 
                        dtype=input.dtype,
                        device=device)),
        dim=1)
    prediction, _ = gen(single_input)
    prediction = torch.clamp(prediction, 0, 1)
    evalDiscr(prediction, 1, 'Prediction:')

