from __future__ import print_function
import argparse
import math
from math import log10
import os
import os.path
from collections import defaultdict
import itertools

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import numpy as np
from torchsummary import summary

from console_progressbar import ProgressBar

#from data import get_training_set, get_test_set
import datasetVideo
import models
import losses
from utils import ScreenSpaceShading

# Training settings
parser = argparse.ArgumentParser(description='Superresolution for Isosurface Raytracing')

parser.add_argument('--dataset', type=str, required=True, help="only 'cloud-video' supported")
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--numberOfImages', type=int, default=-1, help="Number of images taken from the inpt dataset. Default: -1 = unlimited")
parser.add_argument('--useInputDepth', action='store_true', help="Use depth information from the input images?")
parser.add_argument('--useInputNormal', action='store_true', help="Use normal information from the input images?")
parser.add_argument('--deferredShading', action='store_true', help="""
With this option enabled, the network performs superresolution on the unshaded color and normal.
Screen space shading is performed after the superresolution and the loss functions are then applied to those images.
Hence the network only performs a superresolution on the normal + depth + mask
This parameter always implies --useInputDepth and --useInputNormal
""")

parser.add_argument('--restore', type=int, default=-1, help="Restore training from the specified run index")
parser.add_argument('--restoreEpoch', type=int, default=-1, help="In combination with '--restore', specify the epoch from which to recover. Default: last epoch")
parser.add_argument('--pretrained', type=str, default=None, help="Path to a pretrained generator")
parser.add_argument('--pretrainedDiscr', type=str, default=None, help="Path to a pretrained discriminator")

#Model parameters
parser.add_argument('--model', type=str, required=True, help="""
The superresolution model.
Supported nets: 'SubpixelNet', 'EnhanceNet', 'TecoGAN', 'RCAN'
""")
parser.add_argument('--upsample', type=str, default='bilinear', help='Upsampling for EnhanceNet: nearest, bilinear, bicubic, or pixelShuffle')
parser.add_argument('--reconType', type=str, default='residual', help='Block type for EnhanceNet: residual or direct')
parser.add_argument('--useBN', action='store_true', help='Enable batch normalization in the generator and discriminator')
parser.add_argument('--useSN', action='store_true', help='Enable spectral normalization in the generator and discriminator')
parser.add_argument('--numResidualLayers', type=int, default=10, help='Number of residual layers in the generator')
parser.add_argument('--disableTemporal', action='store_true', help='Disables temporal consistency')

#Loss parameters
parser.add_argument('--losses', type=str, required=True, help="""
Comma-separated list of loss functions: mse,perceptual,texture,adv. 
Optinally, the weighting factor can be specified with a colon.
Example: "--losses perceptual:0.1,texture:1e2,adv:10"
""")
parser.add_argument('--perceptualLossLayers', 
                    type=str, 
                     # defaults found with VGGAnalysis.py
                    default='conv_1:0.026423,conv_2:0.009285,conv_3:0.006710,conv_4:0.004898,conv_5:0.003910,conv_6:0.003956,conv_7:0.003813,conv_8:0.002968,conv_9:0.002997,conv_10:0.003631,conv_11:0.004147,conv_12:0.005765,conv_13:0.007442,conv_14:0.009666,conv_15:0.012586,conv_16:0.013377', 
                    help="""
Comma-separated list of layer names for the perceptual loss. 
Note that the convolution layers are numbered sequentially: conv_1, conv2_, ... conv_19.
Optinally, the weighting factor can be specified with a colon: "conv_4:1.0", if omitted, 1 is used.
""")
parser.add_argument('--textureLossLayers', type=str, default='conv_1,conv_3,conv_5', help="""
Comma-separated list of layer names for the perceptual loss. 
Note that the convolution layers are numbered sequentially: conv_1, conv2_, ... conv_19.
Optinally, the weighting factor can be specified with a colon: "conv_4:1.0", if omitted, 1 is used.
""")
parser.add_argument('--discriminator', type=str, default='enhanceNetLarge', help="""
Network architecture for the discriminator.
Possible values: enhanceNetSmall, enhanceNetLarge, tecoGAN
""")
#parser.add_argument('--advDiscrThreshold', type=float, default=None, help="""
#Adverserial training:
#If the cross entropy loss of the discriminator falls below that threshold, the training for the discriminator is stopped.
#Set this to zero to disable the check and use a fixed number of iterations, see --advDiscrMaxSteps, instead.
#""")
parser.add_argument('--advDiscrMaxSteps', type=int, default=2, help="""
Adverserial training:
Maximal number of iterations for the discriminator training.
Set this to -1 to disable the check.
""")
parser.add_argument('--advDiscrInitialSteps', type=int, default=None, help="""
Adverserial training:
Number of iterations for the disciriminator training in the first epoch.
Used in combination with a pretrained generator to let the discriminator catch up.
""")
parser.add_argument('--advDiscrWeightClip', type=float, default=0.01, help="""
For the Wasserstein GAN, this parameter specifies the value of the hyperparameter 'c',
the range in which the discirminator parameters are clipped.
""")
#parser.add_argument('--advGenThreshold', type=float, default=None, help="""
#Adverserial training:
#If the cross entropy loss of the generator falls below that threshold, the training for the generator is stopped.
#Set this to zero to disable the check and use a fixed number of iterations, see --advGenMaxSteps, instead.
#""")
parser.add_argument('--advGenMaxSteps', type=int, default=2, help="""
Adverserial training:
Maximal number of iterations for the generator training.
Set this to -1 to disable the check.
""")
parser.add_argument('--lossBorderPadding', type=int, default=16, help="""
Because flow + warping can't be accurately estimated at the borders of the image,
the border of the input images to the loss (ground truth, low res input, prediction)
are overwritten with zeros. The size of the border is specified by this parameter.
Pass zero to disable this padding. Default=16 as in the TecoGAN paper.
""")

parser.add_argument('--samples', type=int, required=True, help='Number of samples for the train and test dataset')
parser.add_argument('--testFraction', type=float, default=0.2, help='Fraction of test data')
parser.add_argument('--batchSize', type=int, default=64, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=10, help='testing batch size')
parser.add_argument('--testNumFullImages', type=int, default=4, help='number of full size images to test for visualization')
parser.add_argument('--nEpochs', type=int, default=2, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.01')
parser.add_argument('--lrGamma', type=float, default=0.5, help='The learning rate decays every lrStep-epochs by this factor')
parser.add_argument('--lrStep', type=int, default=500, help='The learning rate decays every lrStep-epochs (this parameter) by lrGamma factor')
parser.add_argument('--optim', type=str, default="Adam", help="""
Optimizers. Possible values: RMSprop, Rprop, Adam (default).
""")
parser.add_argument('--noTestImages', action='store_true', help="Don't save full size test images")

parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--seed', type=int, default=124, help='random seed to use. Default=123')
parser.add_argument('--logdir', type=str, default='D:/VolumeSuperResolution/logdir_video', help='directory for tensorboard logs')
parser.add_argument('--modeldir', type=str, default='D:/VolumeSuperResolution/modeldir_video', help='Output directory for the checkpoints')

opt = parser.parse_args()
opt_dict = vars(opt)
opt_dict['unshaded'] = False

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
np.random.seed(opt.seed)

device = torch.device("cuda" if opt.cuda else "cpu")

#########################
# DATASETS + CHANNELS
#########################

print('===> Loading datasets')

if opt.dataset.lower() == 'cloud-video':
    dataset_data = datasetVideo.collect_samples_clouds_video(
        opt.upscale_factor, opt_dict)
else:
    raise ValueError('Unknown dataset %s'%opt.dataset)
print('Dataset input images have %d channels'%dataset_data.input_channels)

input_channels = dataset_data.input_channels
if opt.deferredShading:
    assert(input_channels == 5)
    output_channels = 5 # mask, normalX, normalY, normalZ, depth
    high_res_channels = [3, 4, 5, 6, 7] #rgb at index 0,1,2 ignored
else:
    output_channels = dataset_data.output_channels
    assert output_channels == 3
    high_res_channels = [0, 1, 2] #only rgb
input_channels_with_previous = input_channels + output_channels * (opt.upscale_factor ** 2)

train_set = datasetVideo.DatasetFromSamples(dataset_data, False, opt.testFraction, high_res_channels)
test_set = datasetVideo.DatasetFromSamples(dataset_data, True, opt.testFraction, high_res_channels)
test_full_set = datasetVideo.DatasetFromFullImages(dataset_data, min(opt.testNumFullImages, len(dataset_data.images_low)))
training_data_loader = DataLoader(dataset=train_set, batch_size=opt.batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, batch_size=opt.testBatchSize, shuffle=False)
testing_full_data_loader = DataLoader(dataset=test_full_set, batch_size=1, shuffle=False)

#############################
# MODEL
#############################

print('===> Building model')
model = models.createNetwork(
    opt.model, 
    opt.upscale_factor,
    input_channels_with_previous, 
    high_res_channels,
    output_channels,
    opt)
model.to(device)
print('Model:')
#print(model)
summary(model, 
        input_size=train_set.get_low_res_shape(input_channels_with_previous), 
        device=device.type)

#############################
# SHADING
# for now, only used for testing
#############################
shading = ScreenSpaceShading(device)
shading.fov(30)
shading.ambient_light_color(np.array([0.1,0.1,0.1]))
shading.diffuse_light_color(np.array([1.0, 1.0, 1.0]))
shading.specular_light_color(np.array([0.2, 0.2, 0.2]))
shading.specular_exponent(16)
shading.light_direction(np.array([0.1,0.1,1.0]))
shading.material_color(np.array([1.0, 0.3, 0.0]))

#############################
# LOSSES
#############################

print('===> Building losses')
criterion = losses.LossNet(
    device,
    input_channels, #no previous image, only rgb, mask, optional depth + normal
    output_channels, 
    train_set.get_high_res_shape()[1], #high resolution size
    opt.lossBorderPadding,
    opt)
criterion.to(device)
print('Losses:', criterion)
res = train_set.get_high_res_shape()[1]
criterion.print_summary(
    (output_channels, res, res),
    (output_channels, res, res),
    (input_channels, res, res),
    (output_channels+1, res, res),
    opt.batchSize, device)


#############################
# OPTIMIZER
#############################

print('===> Create Optimizer ('+opt.optim+')')
def createOptimizer(name, parameters, lr):
    if name=='Adam':
        return optim.Adam(parameters, lr=lr)
    elif name=='RMSprop':
        return optim.RMSprop(parameters, lr=lr)
    elif name=='Rprop':
        return optim.Rprop(parameters, lr=lr)
    else:
        raise ValueError("Unknown optimizer "+name)
if criterion.discriminator is None:
    adversarial_training = False
    optimizer = createOptimizer(opt.optim, model.parameters(), lr=opt.lr)
    #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, opt.lrDecay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, opt.lrStep, opt.lrGamma)
else:
    adversarial_training = True
    gen_optimizer = createOptimizer(opt.optim, model.parameters(), lr=opt.lr)
    filter(lambda p: p.requires_grad, criterion.discriminator.parameters())
    discr_optimizer = createOptimizer(
        opt.optim, filter(lambda p: p.requires_grad, criterion.discriminator.parameters()), lr=opt.lr)
    #gen_scheduler = optim.lr_scheduler.ExponentialLR(gen_optimizer, opt.lrDecay)
    #discr_scheduler = optim.lr_scheduler.ExponentialLR(discr_optimizer, opt.lrDecay)
    gen_scheduler = optim.lr_scheduler.StepLR(gen_optimizer, opt.lrStep, opt.lrGamma)
    discr_scheduler = optim.lr_scheduler.StepLR(discr_optimizer, opt.lrStep, opt.lrGamma)

#############################
# PRETRAINED
#############################
if opt.pretrained is not None:
    checkpoint = torch.load(opt.pretrained)
    model.load_state_dict(checkpoint['model'].state_dict())
    #only load the state dict, not the whole model
    #this asserts that the model structure is the same
    print('Using pretrained model for the generator')
if opt.pretrainedDiscr is not None:
    assert criterion.discriminator is not None
    checkpoint = torch.load(opt.pretrainedDiscr)
    criterion.discriminator.load_state_dict(checkpoint['discriminator'])
    print('Using pretrained model for the discriminator')

#############################
# Additional Stuff: Spectral Normalization
# (placed after pretrained, because models without spectral normalization
#  can't be imported as models with normalization
#############################
if opt.useSN:
    from utils.apply_sn import apply_sn
    apply_sn(model)
    if criterion.discriminator is not None:
        apply_sn(criterion.discriminator)
    print("Spectral Normalization applied")

#############################
# OUTPUT DIRECTORIES or RESTORE
#############################

# Run directory
def findNextRunNumber(folder):
    files = os.listdir(folder)
    files = sorted([f for f in files if f.startswith('run')])
    if len(files)==0:
        return 0
    return int(files[-1][3:])

#Check for restoring
startEpoch = 1
if opt.restore == -1:
    nextRunNumber = max(findNextRunNumber(opt.logdir), findNextRunNumber(opt.modeldir)) + 1
else:
    nextRunNumber = opt.restore
    runName = 'run%05d'%nextRunNumber
    modeldir = os.path.join(opt.modeldir, runName)
    if opt.restoreEpoch == -1:
        restoreEpoch = 0
        while True:
            modelInName = os.path.join(modeldir, "model_epoch_{}.pth".format(restoreEpoch+1))
            if not os.path.exists(modelInName):
                break;
            restoreEpoch += 1
    else:
        restoreEpoch = opt.restoreEpoch

    print("Restore training from run", opt.restore,"and epoch",restoreEpoch)
    modelInName = os.path.join(modeldir, "model_epoch_{}.pth".format(restoreEpoch))
    checkpoint = torch.load(modelInName)
    #model.load_state_dict(checkpoint['state_dict'])
    model = checkpoint['model'] #Restore full model
    if adversarial_training:
        criterion.discriminator.load_state_dict(checkpoint['discriminator'])
        discr_optimizer = checkpoint['discr_optimizer']
        gen_optimizer = checkpoint['gen_optimizer']
        discr_scheduler = checkpoint['discr_scheduler']
        gen_scheduler = checkpoint['gen_scheduler']
    else:
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']
    startEpoch = restoreEpoch

#paths
print('Current run: %05d'%nextRunNumber)
runName = 'run%05d'%nextRunNumber
logdir = os.path.join(opt.logdir, runName)
modeldir = os.path.join(opt.modeldir, runName)
if opt.restore == -1:
    os.makedirs(logdir)
    os.makedirs(modeldir)

optStr = str(opt);
print(optStr)
with open(os.path.join(modeldir, 'info.txt'), "w") as text_file:
    text_file.write(optStr)

#tensorboard logger
writer = SummaryWriter(logdir)
writer.add_text('info', optStr, 0)

#############################
# MAIN PART
#############################

def trainNormal(epoch):
    epoch_loss = 0
    scheduler.step()
    num_minibatch = len(training_data_loader)
    pg = ProgressBar(num_minibatch, 'Training', length=50)
    for iteration, batch in enumerate(training_data_loader, 0):
        pg.print_progress_bar(iteration)
        input, flow, target = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        B, _, Cout, Hhigh, Whigh = target.shape
        _, _, Cin, H, W = input.shape
        assert(Cout == output_channels)
        assert(Cin == input_channels)
        assert(H == dataset_data.crop_size)
        assert(W == dataset_data.crop_size)
        assert(Hhigh == dataset_data.crop_size * opt.upscale_factor)
        assert(Whigh == dataset_data.crop_size * opt.upscale_factor)

        optimizer.zero_grad()
        model.train()

        previous_output = None
        loss = 0
        for j in range(dataset_data.num_frames):
            # prepare input
            if j == 0 or opt.disableTemporal:
                previous_warped = torch.zeros(B, Cout, Hhigh, Whigh, 
                                        dtype=input.dtype,
                                        device=device)
                if not opt.deferredShading:
                    previous_warped_loss = torch.cat(
                        (target[:,0,:,:,:],
                         F.interpolate(input[:,j,3:4,:,:], scale_factor=opt.upscale_factor, mode=opt.upsample)),
                         dim=1)
                else:
                    previous_warped_loss = target[:,0,:,:,:]
                # loss takes the ground truth current image as warped previous image,
                # to not introduce a bias and big loss for the first image
            else:
                if not opt.deferredShading:
                    previous_warped_loss = models.VideoTools.warp_upscale(
                        torch.cat(
                            (previous_output,
                             F.interpolate(input[:,j-1,3:4,:,:], scale_factor=opt.upscale_factor, mode=opt.upsample)),
                             dim=1),
                        flow[:, j-1, :, :, :], opt.upscale_factor)
                else:
                    previous_warped_loss = models.VideoTools.warp_upscale(previous_output, flow[:, j-1, :, :, :], opt.upscale_factor)
                previous_warped = previous_warped_loss
            previous_warped_flattened = models.VideoTools.flatten_high(previous_warped, opt.upscale_factor)
            single_input = torch.cat((
                    input[:,j,:,:,:],
                    previous_warped_flattened),
                dim=1)
            # run generator
            prediction, _ = model(single_input)
            prediction = torch.clamp(prediction, 0, 1)
            # evaluate cost
            #loss0,_ = criterion(
            #    target[:,j,:,:,:], 
            #    prediction, 
            #    input[:,j,:,:,:], 
            #    previous_warped_loss)
            #del _
            loss0 = \
                0.001 * F.mse_loss(
                    torch.clamp(target[:,j,0,:,:], 0,1),   \
                    torch.clamp(prediction[:,0,:,:], 0,1)) + \
                F.mse_loss(
                    target[:,j,0:1,:,:] * target[:,j,1:4,:,:],
                    target[:,j,0:1,:,:] * prediction[:,1:4,:,:])
            loss += loss0
            epoch_loss += loss.item()
            # save output
            previous_output = prediction

        loss.backward()
        optimizer.step()
    pg.print_progress_bar(num_minibatch)
    epoch_loss /= num_minibatch * dataset_data.num_frames
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss))
    writer.add_scalar('train/total_loss', epoch_loss, epoch)
    writer.add_scalar('train/lr', scheduler.get_lr()[0], epoch)
        
def trainAdvDiscr(epoch):
    print("===> Epoch %d Training"%epoch)
    discr_scheduler.step()
    writer.add_scalar('train/lr_discr', discr_scheduler.get_lr()[0], epoch)
    num_steps = 0
    mean_gt_logits = 0
    mean_pred_logits = 0
    num_minibatch = len(training_data_loader)

    for p in criterion.discriminator.parameters(): # reset requires_grad
         p.requires_grad = True # they are set to False below in generator update
    model.eval()
    criterion.discriminator.train()

    while True:
        num_steps += 1
        discr_loss = 0
        pg = ProgressBar(num_minibatch, 'Train D (%d)'%num_steps, length=50)
        for iteration, batch in enumerate(training_data_loader):
            pg.print_progress_bar(iteration)
            input, flow, target = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            B, _, Cout, Hhigh, Whigh = target.shape
            _, _, Cin, H, W = input.shape

            #for the Wasserstein GAN, clip the weights
            if criterion.discriminator_clip_weights:
                for p in criterion.discriminator.parameters():
                    p.data.clamp_(-opt.advDiscrWeightClip, +opt.advDiscrWeightClip)

            discr_optimizer.zero_grad()
            previous_output = None
            loss = 0
            gt_logits = 0
            pred_logits = 0
            #iterate over all timesteps
            for j in range(dataset_data.num_frames):
                # prepare input for the generator
                if j == 0 or opt.disableTemporal:
                    previous_warped = torch.zeros(B, Cout, Hhigh, Whigh, 
                                            dtype=input.dtype,
                                            device=device)
                    previous_warped_with_mask = torch.cat(
                        (target[:,0,:,:,:],
                         F.interpolate(input[:,j,3:4,:,:], scale_factor=opt.upscale_factor, mode=opt.upsample)),
                         dim=1)
                else:
                    previous_warped_with_mask = models.VideoTools.warp_upscale(
                        torch.cat(
                            (previous_output,
                             F.interpolate(input[:,j-1,3:4,:,:], scale_factor=opt.upscale_factor, mode=opt.upsample)),
                             dim=1),
                        flow[:, j-1, :, :, :], opt.upscale_factor)
                    previous_warped = previous_warped_with_mask[:,0:3,:,:]
                previous_warped_flattened = models.VideoTools.flatten_high(previous_warped, opt.upscale_factor)
                single_input = torch.cat((
                        input[:,j,:,:,:],
                        previous_warped_flattened),
                    dim=1)
                #evaluate generator
                with torch.no_grad():
                    prediction, _ = model(single_input)
                    prediction = torch.clamp(prediction.detach(), 0, 1)
                #prepare input for the discriminator
                gt_high_with_mask = torch.cat([
                        target[:,j,:,:,:],
                        F.interpolate(input[:,j-1,3:4,:,:], scale_factor=opt.upscale_factor, mode=opt.upsample)],
                    dim=1)
                gt_prev_warped = models.VideoTools.warp_upscale(
                        torch.cat(
                            (target[:,j-1,:,:,:],
                             F.interpolate(input[:,j-1,3:4,:,:], scale_factor=opt.upscale_factor, mode=opt.upsample)),
                             dim=1),
                        flow[:, j-1, :, :, :], opt.upscale_factor)
                prediction_with_mask = torch.cat([prediction, gt_high_with_mask[:,3:4,:,:]], dim=1)
                #evaluate discriminator
                disc_loss, gt_score, pred_score = criterion.train_discriminator(
                    input[:,j,:,:,:], gt_high_with_mask, gt_prev_warped,
                    prediction_with_mask, previous_warped_with_mask)
                #accumulate loss
                loss += disc_loss
                discr_loss += disc_loss.item()
                gt_logits += gt_score
                pred_logits += pred_score
                # save output
                previous_output = prediction
            #train
            loss.backward()
            discr_optimizer.step()
            #test
            mean_gt_logits += gt_logits
            mean_pred_logits += pred_logits
        pg.print_progress_bar(num_minibatch)
        discr_loss /= num_minibatch * dataset_data.num_frames
        mean_gt_logits /= num_minibatch * dataset_data.num_frames
        mean_pred_logits /= num_minibatch * dataset_data.num_frames
        print("---> Train Discriminator {}: loss {:.7f}".format(num_steps, discr_loss))
        print("     mean_gt_score={:.5f}, mean_pred_score={:.5f}".format(mean_gt_logits, mean_pred_logits))
        #break condition
        if opt.advDiscrInitialSteps is not None and epoch==1:
            if num_steps >= opt.advDiscrInitialSteps:
                break
        elif opt.advDiscrMaxSteps and num_steps >= opt.advDiscrMaxSteps:
            break
    writer.add_scalar('train/discr_loss', discr_loss, epoch)
    writer.add_scalar('train/discr_steps', num_steps, epoch)
    print("===> Epoch {} Complete".format(epoch))

def trainAdvGen(epoch):
    print("===> Epoch %d Training"%epoch)
    gen_scheduler.step()
    writer.add_scalar('train/lr_gen', gen_scheduler.get_lr()[0], epoch)
    num_steps = 0
    mean_gt_logits = 0
    mean_pred_logits = 0
    num_minibatch = len(training_data_loader)

    for p in criterion.discriminator.parameters():
         p.requires_grad = False # to avoid computation
    model.train()
    criterion.discriminator.eval()

    while True:
        num_steps += 1
        epoch_loss = 0
        discr_loss = 0
        pg = ProgressBar(num_minibatch, 'Train G (%d)'%num_steps, length=50)
        for iteration, batch in enumerate(training_data_loader):
            pg.print_progress_bar(iteration)
            input, flow, target = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            B, _, Cout, Hhigh, Whigh = target.shape
            _, _, Cin, H, W = input.shape
            #train generator
            gen_optimizer.zero_grad()
            loss = 0
            previous_output = None
            #iterate over all timesteps
            for j in range(dataset_data.num_frames):
                # prepare input
                if j == 0 or opt.disableTemporal:
                    previous_warped = torch.zeros(B, Cout, Hhigh, Whigh, 
                                            dtype=input.dtype,
                                            device=device)
                    previous_warped_with_mask = torch.cat(
                        (target[:,0,:,:,:],
                         F.interpolate(input[:,j,3:4,:,:], scale_factor=opt.upscale_factor, mode=opt.upsample)),
                         dim=1)
                    # loss takes the ground truth current image as warped previous image,
                    # to not introduce a bias and big loss for the first image
                else:
                    previous_warped_with_mask = models.VideoTools.warp_upscale(
                        torch.cat(
                            (previous_output,
                             F.interpolate(input[:,j-1,3:4,:,:], scale_factor=opt.upscale_factor, mode=opt.upsample)),
                             dim=1),
                        flow[:, j-1, :, :, :], opt.upscale_factor)
                    previous_warped = previous_warped_with_mask[:,0:3,:,:]
                previous_warped_flattened = models.VideoTools.flatten_high(previous_warped, opt.upscale_factor)
                single_input = torch.cat((
                        input[:,j,:,:,:],
                        previous_warped_flattened),
                    dim=1)
                #evaluate generator
                prediction, _ = model(single_input)
                prediction = torch.clamp(prediction, 0, 1)
                #evaluate loss
                loss0, map = criterion(
                    target[:,j,:,:,:], 
                    prediction, 
                    input[:,j,:,:,:],
                    previous_warped_with_mask)
                loss += loss0
                epoch_loss += loss0.item()
                # save output
                previous_output = prediction
            #evaluate gradients
            loss.backward()
            gen_optimizer.step()
        pg.print_progress_bar(num_minibatch)
        epoch_loss /= num_minibatch * dataset_data.num_frames
        print("---> Train Generator {}: loss {:.7f}".format(num_steps, epoch_loss))
        if opt.advGenMaxSteps > 0 and num_steps >= opt.advGenMaxSteps:
            break

    writer.add_scalar('train/gen_loss', discr_loss, epoch)
    writer.add_scalar('train/gen_steps', num_steps, epoch)
    print("===> Epoch {} Complete".format(epoch))

def trainAdv_v2(epoch):
    """
    Second version of adverserial training, 
    for each batch, train both discriminator and generator.
    Not full epoch for each seperately
    """
    print("===> Epoch %d Training"%epoch)
    discr_scheduler.step()
    writer.add_scalar('train/lr_discr', discr_scheduler.get_lr()[0], epoch)
    gen_scheduler.step()
    writer.add_scalar('train/lr_gen', gen_scheduler.get_lr()[0], epoch)

    disc_steps = opt.advDiscrInitialSteps if opt.advDiscrInitialSteps is not None and epoch==1 else opt.advDiscrMaxSteps
    gen_steps = opt.advGenMaxSteps

    num_minibatch = len(training_data_loader)
    model.train()
    criterion.discriminator.train()

    total_discr_loss = 0
    total_gen_loss = 0

    pg = ProgressBar(num_minibatch, 'Train', length=50)
    for iteration, batch in enumerate(training_data_loader):
        pg.print_progress_bar(iteration)
        input, flow, target = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        B, _, Cout, Hhigh, Whigh = target.shape
        _, _, Cin, H, W = input.shape

        # DISCRIMINATOR
        for _ in range(disc_steps):
            discr_optimizer.zero_grad()
            gen_optimizer.zero_grad()
            loss = 0
            #iterate over all timesteps
            for j in range(dataset_data.num_frames):
                # prepare input for the generator
                if j == 0 or opt.disableTemporal:
                    previous_warped = torch.zeros(B, Cout, Hhigh, Whigh, 
                                            dtype=input.dtype,
                                            device=device)
                    previous_warped_with_mask = torch.cat(
                        (target[:,0,:,:,:],
                         F.interpolate(input[:,j,3:4,:,:], scale_factor=opt.upscale_factor, mode=opt.upsample)),
                         dim=1)
                else:
                    if not opt.deferredShading:
                        previous_warped_with_mask = models.VideoTools.warp_upscale(
                            torch.cat(
                                (previous_output,
                                 F.interpolate(input[:,j-1,3:4,:,:], scale_factor=opt.upscale_factor, mode=opt.upsample)),
                                 dim=1),
                            flow[:, j-1, :, :, :], opt.upscale_factor)
                    else:
                        previous_warped_with_mask = models.VideoTools.warp_upscale(previous_output, flow[:, j-1, :, :, :], opt.upscale_factor)
                    previous_warped = previous_warped_with_mask[:,0:3,:,:]
                previous_warped_flattened = models.VideoTools.flatten_high(previous_warped, opt.upscale_factor)
                single_input = torch.cat((
                        input[:,j,:,:,:],
                        previous_warped_flattened),
                    dim=1)
                #evaluate generator
                with torch.no_grad():
                    prediction, _ = model(single_input)
                    prediction = torch.clamp(prediction.detach(), 0, 1)
                #prepare input for the discriminator
                if not opt.deferredShading:
                    gt_high_with_mask = torch.cat([
                        target[:,j,:,:,:],
                        F.interpolate(input[:,j-1,3:4,:,:], scale_factor=opt.upscale_factor, mode=opt.upsample)],
                    dim=1)
                    gt_prev_warped = models.VideoTools.warp_upscale(
                            torch.cat(
                                (target[:,j-1,:,:,:],
                                 F.interpolate(input[:,j-1,3:4,:,:], scale_factor=opt.upscale_factor, mode=opt.upsample)),
                                 dim=1),
                            flow[:, j-1, :, :, :], opt.upscale_factor)
                    prediction_with_mask = torch.cat([prediction, gt_high_with_mask[:,3:4,:,:]], dim=1)
                else:
                    gt_prev_warped = models.VideoTools.warp_upscale(target[:,j-1,:,:,:], flow[:, j-1, :, :, :], opt.upscale_factor)
                    prediction_with_mask = prediction
                #evaluate discriminator
                disc_loss, gt_score, pred_score = criterion.train_discriminator(
                    input[:,j,:,:,:], gt_high_with_mask, gt_prev_warped,
                    prediction_with_mask, previous_warped_with_mask)

                loss += disc_loss
                # save output
                previous_output = prediction
            loss.backward()
            discr_optimizer.step()
        total_discr_loss += loss.item()

        # GENERATOR
        for _ in range(disc_steps):
            discr_optimizer.zero_grad()
            gen_optimizer.zero_grad()
            loss = 0
            #iterate over all timesteps
            for j in range(dataset_data.num_frames):
                # prepare input for the generator
                if j == 0 or opt.disableTemporal:
                    previous_warped = torch.zeros(B, Cout, Hhigh, Whigh, 
                                            dtype=input.dtype,
                                            device=device)
                    previous_warped_with_mask = torch.cat(
                        (target[:,0,:,:,:],
                         F.interpolate(input[:,j,3:4,:,:], scale_factor=opt.upscale_factor, mode=opt.upsample)),
                         dim=1)
                else:
                    if not opt.deferredShading:
                        previous_warped_with_mask = models.VideoTools.warp_upscale(
                            torch.cat(
                                (previous_output,
                                 F.interpolate(input[:,j-1,3:4,:,:], scale_factor=opt.upscale_factor, mode=opt.upsample)),
                                 dim=1),
                            flow[:, j-1, :, :, :], opt.upscale_factor)
                    else:
                        previous_warped_with_mask = models.VideoTools.warp_upscale(previous_output, flow[:, j-1, :, :, :], opt.upscale_factor)
                    previous_warped = previous_warped_with_mask[:,0:3,:,:]
                previous_warped_flattened = models.VideoTools.flatten_high(previous_warped, opt.upscale_factor)
                single_input = torch.cat((
                        input[:,j,:,:,:],
                        previous_warped_flattened),
                    dim=1)
                #evaluate generator
                prediction, _ = model(single_input)
                prediction = torch.clamp(prediction, 0, 1)
                #evaluate loss
                loss0, map = criterion(
                    target[:,j,:,:,:], 
                    prediction, 
                    input[:,j,:,:,:],
                    previous_warped_with_mask)
                loss += loss0
                # save output
                previous_output = prediction
            loss.backward()
            gen_optimizer.step()
        total_gen_loss += loss.item()
    pg.print_progress_bar(num_minibatch)

    total_discr_loss /= num_minibatch * dataset_data.num_frames
    total_gen_loss /= num_minibatch * dataset_data.num_frames

    writer.add_scalar('train/discr_loss', total_discr_loss, epoch)
    writer.add_scalar('train/gen_loss', total_gen_loss, epoch)
    print("===> Epoch {} Complete".format(epoch))

def test(epoch):
    avg_psnr = 0
    avg_losses = defaultdict(float)
    with torch.no_grad():
        num_minibatch = len(testing_data_loader)
        pg = ProgressBar(num_minibatch, 'Testing', length=50)
        model.eval()
        if criterion.discriminator is not None:
            criterion.discriminator.eval()
        for iteration, batch in enumerate(testing_data_loader, 0):
            pg.print_progress_bar(iteration)
            input, flow, target = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            B, _, Cout, Hhigh, Whigh = target.shape
            _, _, Cin, H, W = input.shape

            previous_output = None
            for j in range(dataset_data.num_frames):
                # prepare input
                if j == 0 or opt.disableTemporal:
                    previous_warped = torch.zeros(B, Cout, Hhigh, Whigh, 
                                            dtype=input.dtype,
                                            device=device)
                    if not opt.deferredShading:
                        previous_warped_loss = torch.cat(
                            (target[:,0,:,:,:],
                             F.interpolate(input[:,j,3:4,:,:], scale_factor=opt.upscale_factor, mode=opt.upsample)),
                             dim=1)
                    else:
                        previous_warped_loss = target[:,0,:,:,:]
                    # loss takes the ground truth current image as warped previous image,
                    # to not introduce a bias and big loss for the first image
                else:
                    if not opt.deferredShading:
                        previous_warped_loss = models.VideoTools.warp_upscale(
                            torch.cat(
                                (previous_output,
                                 F.interpolate(input[:,j-1,3:4,:,:], scale_factor=opt.upscale_factor, mode=opt.upsample)),
                                 dim=1),
                            flow[:, j-1, :, :, :], opt.upscale_factor)
                    else:
                        previous_warped_with_mask = models.VideoTools.warp_upscale(previous_output, flow[:, j-1, :, :, :], opt.upscale_factor)
                    previous_warped = previous_warped_loss
                previous_warped_flattened = models.VideoTools.flatten_high(previous_warped, opt.upscale_factor)
                single_input = torch.cat((
                        input[:,j,:,:,:],
                        previous_warped_flattened),
                    dim=1)
                # run generator
                prediction, _ = model(single_input)
                prediction = torch.clamp(prediction, 0, 1)
                # evaluate cost
                loss,loss_values = criterion(target[:,j,:,:,:], prediction, input[:,j,:,:,:], previous_warped_loss)
                avg_losses['total_loss'] += loss.item()
                psnr = 10 * log10(1 / loss_values['mse'])
                avg_losses['psnr'] += psnr
                for key, value in loss_values.items():
                    avg_losses[key] += value
                # extra: evaluate discriminator on ground truth data
                if criterion.discriminator is not None:
                    input_high = F.interpolate(input[:,j,:,:,:], size=(Hhigh, Whigh), mode=opt.upsample)
                    gt_high_with_mask = torch.cat([target[:,j,:,:,:], input_high[:,3:4,:,:]], dim=1)
                    if criterion.discriminator_use_previous_image:
                        gt_prev_warped = models.VideoTools.warp_upscale(
                            torch.cat(
                                (target[:,j-1,:,:,:],
                                 F.interpolate(input[:,j-1,3:4,:,:], scale_factor=opt.upscale_factor, mode=opt.upsample)),
                                 dim=1),
                            flow[:, j-1, :, :, :], opt.upscale_factor)
                        input_images = torch.cat([input_high, gt_high_with_mask, gt_prev_warped], dim=1)
                    else:
                        input_images = torch.cat([input_high, gt_high_with_mask], dim=1)
                    input_images = losses.LossNet.pad(input_images, criterion.padding)
                    discr_gt = criterion.adv_loss(criterion.discriminator(input_images))
                    avg_losses['discr_gt'] += discr_gt.item()

                # save output for next frame
                previous_output = prediction
        pg.print_progress_bar(num_minibatch)
    for key in avg_losses.keys():
        avg_losses[key] /= num_minibatch * dataset_data.num_frames
    print("===> Avg. PSNR: {:.4f} dB".format(avg_losses['psnr']))
    print("  losses:",avg_losses)
    for key, value in avg_losses.items():
        writer.add_scalar('test/%s'%key, value, epoch)

def test_images(epoch):
    def write_image(img, filename):
        out_img = img.cpu().detach().numpy()
        out_img *= 255.0
        out_img = out_img.clip(0, 255)
        out_img = np.uint8(out_img)
        writer.add_image(filename, out_img, epoch)

    with torch.no_grad():
        num_minibatch = len(testing_full_data_loader)
        pg = ProgressBar(num_minibatch, 'Test %d Images'%num_minibatch, length=50)
        model.eval()
        if criterion.discriminator is not None:
            criterion.discriminator.eval()
        for i,batch in enumerate(testing_full_data_loader):
            pg.print_progress_bar(i)
            input, flow = batch[0].to(device), batch[1].to(device)
            B, _, Cin, H, W = input.shape
            Hhigh = H * opt.upscale_factor
            Whigh = W * opt.upscale_factor
            Cout = output_channels

            if opt.deferredShading:
                channel_mask = [1, 2, 3] #normal
            else:
                channel_mask = [0, 1, 2] #rgb

            previous_output = None
            for j in range(dataset_data.num_frames):
                # prepare input
                if j == 0 or opt.disableTemporal:
                    previous_warped = torch.zeros(B, Cout, Hhigh, Whigh, 
                                         dtype=input.dtype,
                                         device=device)
                else:
                    previous_warped = models.VideoTools.warp_upscale(previous_output, flow[:, j-1, :, :, :], opt.upscale_factor)
                previous_warped_flattened = models.VideoTools.flatten_high(previous_warped, opt.upscale_factor)
                single_input = torch.cat((
                        input[:,j,:,:,:],
                        previous_warped_flattened),
                    dim=1)
                # write warped previous frame
                write_image(previous_warped[0, channel_mask], 'image%03d/frame%03d_warped' % (i, j))
                # run generator and cost
                prediction, residual = model(single_input)
                prediction = torch.clamp(prediction, 0, 1)
                # write prediction image
                write_image(prediction[0, channel_mask], 'image%03d/frame%03d_prediction' % (i, j))
                # write residual image
                if residual is not None:
                    write_image(residual[0, channel_mask], 'image%03d/frame%03d_residual' % (i, j))
                # write shaded image if network runs in deferredShading mode
                if opt.deferredShading:
                    shaded_image = shading(prediction)
                    write_image(shaded_image[0], 'image%03d/frame%03d_shaded' % (i, j))
                # save output for next frame
                previous_output = prediction
        pg.print_progress_bar(num_minibatch)

    print("Test images sent to Tensorboard for visualization")

def checkpoint(epoch):
    model_out_path = os.path.join(modeldir, "model_epoch_{}.pth".format(epoch))
    state = {'epoch': epoch + 1, 'model': model, 'parameters':opt_dict}
    if not adversarial_training:
        state.update({'optimizer':optimizer, 'scheduler':scheduler})
    else:
        state.update({'discr_optimizer':discr_optimizer, 
                      'gen_optimizer':gen_optimizer,
                      'discr_scheduler':discr_scheduler,
                      'gen_scheduler':gen_scheduler,
                      'discriminator': criterion.discriminator.state_dict()})
    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

if not os.path.exists(opt.modeldir):
    os.mkdir(opt.modeldir)
if not os.path.exists(opt.logdir):
    os.mkdir(opt.logdir)

print('===> Start Training')
if not adversarial_training:
    test_images(0)
    for epoch in range(startEpoch, opt.nEpochs + 1):
        trainNormal(epoch)
        test(epoch)
        if (epoch < 20 or (epoch%10==0)) and not opt.noTestImages:
            test_images(epoch)
        checkpoint(epoch)
else:
    test(0)
    if not opt.noTestImages:
        test_images(0)
    for epoch in range(startEpoch, opt.nEpochs + 1):
        trainAdv_v2(epoch)
        test(epoch)
        if (epoch < 20 or (epoch%10==0)) and not opt.noTestImages:
            test_images(epoch)
        checkpoint(epoch)

    #for epoch in range(startEpoch, opt.nEpochs + 1, 2):
    #    trainAdvDiscr(epoch)
    #    test(epoch)
    #    checkpoint(epoch)
    #    trainAdvGen(epoch+1)
    #    test(epoch+1)
    #    if (epoch < 20 or (epoch%10==9)) and not opt.noTestImages:
    #        test_images(epoch+1)
    #    checkpoint(epoch+1)

#writer.export_scalars_to_json(os.path.join(opt.logdir, "all_scalars.json"))
writer.close()