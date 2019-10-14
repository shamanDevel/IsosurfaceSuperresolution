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

import inference
from utils import ScreenSpaceShading, MeanVariance, MSSSIM

########################################
# CONFIGURATION
########################################

#input
RENDERER_CPU = '../bin/CPURenderer.exe'
RENDERER_GPU = '../bin/GPURenderer.exe'
MODEL_DIR = "D:/VolumeSuperResolution"
DATA_DIR_CPU = "../../data/"
DATA_DIR_GPU = "../../data/"


DATASETS = [
    #{
    #    'file':'clouds/inputVBX/cloud-049.vbx',
    #    'name':'Cloud-049',
    #    'iso':0.3,
    #    'material':[255, 255, 255],
    #    'ambient':[25, 25, 25],
    #    'diffuse':[250, 250, 250],
    #    'specular':[35, 35, 35],
    #    'distance':1.6,
    #    'lookAt': [0,0,0]
    #},
    {
        #'file':'volumes/vbx/snapshot_272_1024.vbx',
        'file':'volumes/vdb/snapshot_272_1024.vdb',
        'name':'Ejecta-1024',
        'iso':0.34,
        'material':[255, 255, 255],
        'ambient':[25, 25, 25],
        'diffuse':[172, 177, 179],
        'specular':[35, 35, 35],
        'distance':2.0,
        'orientation': inference.Orientation.Yp,
        'pitch': 0.38,
        'lookAt': [0,0,0]
    },
    #{
    #    'file':'volumes/vbx/ppmt273_512_border.vbx',
    #    'name':'Richtmyer-Meshkov-512',
    #    'iso':0.34,
    #    'material':[255, 255, 255],
    #    'ambient':[25, 25, 25],
    #    'diffuse':[172, 177, 179],
    #    'specular':[35, 35, 35],
    #    'distance':1.12,
    #    'orientation': inference.Orientation.Zm,
    #    'pitch': 0.522,
    #    'lookAt': [0,0,-0.07]
    #},
    {
        'file':'volumes/vbx/ppmt273_1024_border.vbx',
        'name':'Richtmyer-Meshkov-1024',
        'iso':0.34,
        'material':[255, 255, 255],
        'ambient':[25, 25, 25],
        'diffuse':[172, 177, 179],
        'specular':[35, 35, 35],
        'distance':1.12,
        'orientation': inference.Orientation.Zm,
        'pitch': 0.522,
        'lookAt': [0,0,-0.07]
    },
    {
        'file':'volumes/vbx/vmhead256cubed.vbx',
        'name':'vmhead256cubed',
        'iso':0.31,
        'material':[255, 255, 255],
        'ambient':[25, 25, 25],
        'diffuse':[172, 177, 179],
        'specular':[35, 35, 35],
        'distance':2.4,
        'orientation': inference.Orientation.Zm,
        'pitch': 0.0,
        'lookAt': [0,0,0]
    },
    {
        'file':'volumes/vbx/cleveland70.vbx',
        'name':'cleveland70',
        'iso':0.02,
        'material':[255, 255, 255],
        'ambient':[25, 25, 25],
        'diffuse':[172, 177, 179],
        'specular':[35, 35, 35],
        'distance':1.5,
        'orientation': inference.Orientation.Zm,
        'pitch': 0.6 ,
        'lookAt': [0,0,0]
    },
    ]

MODEL_NEAREST = "nearest"
MODEL_BILINEAR = "bilinear"
MODEL_BICUBIC = "bicubic"
MODELS = [
    {'name' : MODEL_NEAREST, 'path' : None},
    {'name' : MODEL_BILINEAR, 'path' : None},
    {'name' : MODEL_BICUBIC, 'path' : None},
    {
        'name': 'gen_l1normal',
        'path': 'pretrained_unshaded/gen_l1normal.pth',
        'temporal': True,
        'masking': True
    },
    {
        'name': 'gen_gan_1',
        'path': 'pretrained_unshaded/gen_gan_1.pth',
        'temporal': True,
        'masking': True
    },
    {
        'name': 'gen_gan_2',
        'path': 'pretrained_unshaded/gen_gan_2b_nAO_wCol.pth',
        'temporal': True,
        'masking': True
    },
    ]
UPSCALING = 4

AO_SAMPLES = 0 #128
AO_RADIUS = 0.05
AO_STRENGTH = 0.0 #0.8
SPECULAR_EXPONENT = 8

#rendering
OUTPUT_FOLDER = 'D:\\VolumeSuperResolution\\PSNR'
SPHERE_SAMPLES = 50 #100
ROTATION_SAMPLES = 6
BACKGROUND = [0,0,0] #[1,1,1]
RESOLUTION = (1000, 1000)
RANDOM_SEED = 42

# Load models
device = torch.device("cuda")
models = [None]*len(MODELS)
for i,m in enumerate(MODELS):
    p = m['path']
    if p is not None:
        models[i] = inference.LoadedModel(os.path.join(MODEL_DIR,p), device, UPSCALING)

# create output folder
os.makedirs(OUTPUT_FOLDER, exist_ok = True)

def getRandomPointOnSphere(dist : float):
    pos = np.random.rand(3)
    pos = pos / np.linalg.norm(pos) * dist
    return pos

# Render each dataset
for i in range(len(DATASETS)):
    #output
    outputFolder = os.path.join(OUTPUT_FOLDER, DATASETS[i]['name'])
    os.makedirs(outputFolder, exist_ok = True)
    statistics = open(os.path.join(outputFolder, "AA_AllFrames.txt"), "w")
    statistics.write("Origin\tRotation\t" + 
                     "\t".join((m['name']+"-PSNR-normal\t"+m['name']+"-PSNR-color\t"+m['name']+"-SSIM-normal\t"+m['name']+"-SSIM-color") for m in MODELS) + 
                     "\n")
    # create renderer
    material = inference.Material(DATASETS[i]['iso'])
    renderer_path = RENDERER_CPU if DATASETS[i]['file'].endswith('vdb') else RENDERER_GPU
    data_dir = DATA_DIR_CPU if DATASETS[i]['file'].endswith('vdb') else DATA_DIR_GPU
    datasetfile = os.path.join(data_dir, DATASETS[i]['file'])
    print('Open', datasetfile)
    renderer = inference.Renderer(renderer_path, datasetfile, material, inference.Camera(RESOLUTION[0], RESOLUTION[1]))
    time.sleep(5)
    renderer.send_command("aoradius=%5.3f\n"%float(AO_RADIUS))
    # create shading
    shading = ScreenSpaceShading(torch.device('cpu'))
    shading.fov(30)
    shading.light_direction(np.array([0.1,0.1,1.0]))
    shading.ambient_light_color(np.array(DATASETS[i]['ambient'])/255.0)
    shading.diffuse_light_color(np.array(DATASETS[i]['diffuse'])/255.0)
    shading.specular_light_color(np.array(DATASETS[i]['specular'])/255.0)
    shading.specular_exponent(SPECULAR_EXPONENT)
    shading.material_color(np.array(DATASETS[i]['material'])/255.0)
    shading.ambient_occlusion(AO_STRENGTH)
    shading.background(np.array(BACKGROUND))

    # prepare running stats
    numModels = len(MODELS)

    mvPSNRColor = [MeanVariance() for i in range(numModels)]
    mvPSNRNormal = [MeanVariance() for i in range(numModels)]
    minPSNR = [float("inf")] * numModels
    minPSNRIndex = [None] * numModels
    maxPSNR = [0] * numModels
    maxPSNRIndex = [None] * numModels

    mvSSIMColor = [MeanVariance() for i in range(numModels)]
    mvSSIMNormal = [MeanVariance() for i in range(numModels)]
    minSSIM = [float("inf")] * numModels
    minSSIMIndex = [None] * numModels
    maxSSIM = [0] * numModels
    maxSSIMIndex = [None] * numModels

    # prepare SSIM
    ssimLoss = MSSSIM()
    ssimLoss.to(device)
    def computeSSIM(npArray1, npArray2):
        torchArray1 = torch.unsqueeze(torch.from_numpy(npArray1).to(device), 0)
        torchArray2 = torch.unsqueeze(torch.from_numpy(npArray2).to(device), 0)
        return ssimLoss(torchArray1, torchArray2).item()

    breakAll = False
    try:
        # statistics
        pg = ProgressBar(SPHERE_SAMPLES * ROTATION_SAMPLES, 'Render', length=50)
        np.random.seed(RANDOM_SEED)
        for j1 in range(SPHERE_SAMPLES):
            currentOrigin = getRandomPointOnSphere(DATASETS[i]['distance']) + np.array(DATASETS[i]['lookAt'])
            for j2 in range(ROTATION_SAMPLES):
                statistics.write("%d\t%d" % (j1, j2))
                currentLookAt = np.array(DATASETS[i]['lookAt'])
                currentUp = getRandomPointOnSphere(1.0)
                pg.print_progress_bar(j1 * ROTATION_SAMPLES + j2)

                # set rendering parameters (no AO)
                renderer.send_command("aosamples=0\n")
                renderer.send_command("cameraOrigin=%5.3f,%5.3f,%5.3f\n"%(currentOrigin[0], currentOrigin[1], currentOrigin[2]))
                renderer.send_command("cameraLookAt=%5.3f,%5.3f,%5.3f\n"%(currentLookAt[0], currentLookAt[1], currentLookAt[2]))
                renderer.send_command("cameraUp=%5.3f,%5.3f,%5.3f\n"%(currentUp[0], currentUp[1], currentUp[2]))

                # render ground truth
                renderer.send_command("resolution=%d,%d\n"%(RESOLUTION[0], RESOLUTION[1]))
                renderer.send_command("render\n")
                gt_image = renderer.read_image(RESOLUTION[0], RESOLUTION[1])
                gt_mask = gt_image[3:4,:,:] > 0.5
                gt_image = np.concatenate((
                    gt_image[0:3,:,:],
                    gt_image[3:4,:,:]*2-1, #transform mask into -1,+1
                    gt_image[4: ,:,:]), axis=0)
                gt_image_shaded_input = np.concatenate((gt_image[3:4,:,:], gt_image[4:8,:,:], gt_image[10:11,:,:]), axis=0)
                gt_image_shaded = torch.clamp(shading(torch.unsqueeze(torch.from_numpy(gt_image_shaded_input),0)), 0, 1).numpy()[0]
                gt_image[0:3,:,:] = gt_image_shaded
                # and save
                #filename = "frame_%d_%d_gt.png" % (j1, j2)
                #scipy.misc.imsave(os.path.join(outputFolder, filename), gt_image[0:3,:,:].transpose((2,1,0)))
                # Debug: NaN
                for x in range(gt_image.shape[1]):
                    for y in range(gt_image.shape[2]):
                        if np.any(np.isnan(gt_image[:,x,y])):
                            print("NaN at %d,%d:" %(x,y), gt_image[:,x,y])

                # render each model
                for k,m in enumerate(MODELS):
            
                    #print('Render', m['name'])
                    p = m['name']

                    w2 = int(RESOLUTION[0] // UPSCALING)
                    h2 = int(RESOLUTION[1] // UPSCALING)
                    renderer.send_command("resolution=%d,%d\n"%(w2, h2))

                    # render and read back
                    renderer.send_command("render\n")
                    image = renderer.read_image(w2, h2)
                    # Debug: NaN
                    for x in range(image.shape[1]):
                        for y in range(image.shape[2]):
                            if np.any(np.isnan(image[:,x,y])):
                                print("NaN at %d,%d:" %(x,y), image[:,x,y])
                    # preprocess
                    original_image = np.copy(image)
                    image = np.concatenate((
                        image[0:3,:,:],
                        image[3:4,:,:]*2-1, #transform mask into -1,+1
                        image[4: ,:,:]), axis=0)
                    image_shaded_input = np.concatenate((image[3:4,:,:], image[4:8,:,:], image[10:11,:,:]), axis=0)
                    image_shaded = torch.clamp(shading(torch.unsqueeze(torch.from_numpy(image_shaded_input),0)), 0, 1).numpy()[0]
                    image[0:3,:,:] = image_shaded
                    # compute color
                    if p==MODEL_NEAREST:
                        image = cv.resize(image.transpose((2, 1, 0)),
                                            dsize=None,
                                            fx=UPSCALING, 
                                            fy=UPSCALING, 
                                            interpolation=cv.INTER_NEAREST)
                        image = image.transpose((2, 1, 0))
                    elif p==MODEL_BILINEAR:
                        image = cv.resize(image.transpose((2, 1, 0)),
                                            dsize=None,
                                            fx=UPSCALING, 
                                            fy=UPSCALING, 
                                            interpolation=cv.INTER_LINEAR)
                        image = image.transpose((2, 1, 0))
                    elif p==MODEL_BICUBIC:
                        image = cv.resize(image.transpose((2, 1, 0)),
                                            dsize=None,
                                            fx=UPSCALING, 
                                            fy=UPSCALING, 
                                            interpolation=cv.INTER_CUBIC)
                        image = image.transpose((2, 1, 0))
                    else: # network
                        previous_image = None
                        imageInput = np.copy(image)
                        # unshaded input
                        imageInput = torch.unsqueeze(torch.from_numpy(imageInput), dim=0).to(device)
                        imageRaw = models[k].inference(imageInput, previous_image)

                        imageRaw = torch.cat([
                            torch.clamp(imageRaw[:,0:1,:,:], -1, +1),
                            ScreenSpaceShading.normalize(imageRaw[:,1:4,:,:], dim=1),
                            torch.clamp(imageRaw[:,4:,:,:], 0, 1)
                            ], dim=1)
                        previous_image = imageRaw
                    
                        image = image.transpose((2, 1, 0))
                        image = cv.resize(image, dsize=None, fx=UPSCALING, fy=UPSCALING, interpolation=cv.INTER_LINEAR)
                        image = image.transpose((2, 1, 0))
                        base_mask = np.copy(image[3,:,:])
                        imageRawCpu = imageRaw.cpu()
                        image[0:3,:,:] = np.clip(shading(imageRawCpu)[0].numpy(), 0, 1)
                        imageRawCpu = imageRawCpu[0].numpy()
                        #image[[3,4,5,6,7,10],:,:] = imageRawCpu
                        image[4:7,:,:] = imageRawCpu[1:4,:,:]

                        if m['masking']:
                            mask = (base_mask*0.5+0.5)
                            image = BACKGROUND[0] + mask[np.newaxis,:,:] * (image - BACKGROUND[0])

                    image[0:3,:,:] = np.clip(image[0:3,:,:], 0.0, 1.0)

                    #compute PSNR
                    #TODO: masking does not work, psnrNormal is always the same for superres
                    #How can I do it without numpy's masking?
                    gt_mask = np.broadcast_to(gt_mask, gt_image.shape)
                    gt_image_masked = gt_image #np.ma.array(gt_image, mask=gt_mask)
                    image_masked = image #np.ma.array(image, mask=gt_mask)
                    psnrNormal = 10 * math.log10(1 / ((gt_image_masked[4:7,:,:]-image_masked[4:7,:,:])**2).mean(axis=None))
                    psnrColor = 10 * math.log10(1 / ((gt_image_masked[0:3,:,:]-image_masked[0:3,:,:])**2).mean(axis=None))
                    statistics.write("\t%e\t%e"%(psnrNormal, psnrColor))
                    #print("Model %d:\t%e\t%e"%(k,psnrNormal, psnrColor))
                    if math.isnan(psnrNormal) or math.isnan(psnrColor):
                        print("NaN at %d,%d, %s" %(j1, j2, p))
                        filename = "ZZ_NaN_%d_%d_%s.png" % (j1, j2, p)
                        scipy.misc.imsave(os.path.join(outputFolder, filename), np.nan_to_num(image[0:3,:,:].transpose((2,1,0))))
                        continue
                    mvPSNRNormal[k].append(psnrNormal)
                    mvPSNRColor[k].append(psnrColor)
                    #min and max
                    if psnrColor > maxPSNR[k]:
                        maxPSNR[k] = psnrColor
                        maxPSNRIndex[k] = (j1, j2)
                        filename = "%s_maxPSNR.png" % p
                        scipy.misc.imsave(os.path.join(outputFolder, filename), np.nan_to_num(image[0:3,:,:].transpose((2,1,0))))
                        filename = "%s_maxPSNR_gt.png" % p
                        scipy.misc.imsave(os.path.join(outputFolder, filename), np.nan_to_num(gt_image[0:3,:,:].transpose((2,1,0))))
                    if psnrColor < minPSNR[k]:
                        minPSNR[k] = psnrColor
                        minPSNRIndex[k] = (j1, j2)
                        filename = "%s_minPSNR.png" % p
                        scipy.misc.imsave(os.path.join(outputFolder, filename), np.nan_to_num(image[0:3,:,:].transpose((2,1,0))))
                        filename = "%s_minPSNR_gt.png" % p
                        scipy.misc.imsave(os.path.join(outputFolder, filename), np.nan_to_num(gt_image[0:3,:,:].transpose((2,1,0))))

                    #compute SSIM
                    #Based on:
                    # Image Quality Assessment: From Error Visibility to Structural Similarity, Zhou Wang et.al.
                    # https://github.com/jorge-pessoa/pytorch-msssim/blob/master/pytorch_msssim/__init__.py
                    ssimNormal = computeSSIM(gt_image_masked[4:7,:,:], image_masked[4:7,:,:])
                    ssimColor = computeSSIM(gt_image_masked[0:3,:,:], image_masked[0:3,:,:])
                    statistics.write("\t%e\t%e"%(ssimNormal, ssimColor))
                    if math.isnan(ssimNormal) or math.isnan(ssimColor):
                        print("NaN at %d,%d, %s" %(j1, j2, p))
                        filename = "ZZ_NaN_%d_%d_%s.png" % (j1, j2, p)
                        scipy.misc.imsave(os.path.join(outputFolder, filename), np.nan_to_num(image[0:3,:,:].transpose((2,1,0))))
                        continue
                    mvSSIMNormal[k].append(ssimNormal)
                    mvSSIMColor[k].append(ssimColor)
                    #min and max
                    if ssimColor > maxSSIM[k]:
                        maxSSIM[k] = ssimColor
                        maxSSIMIndex[k] = (j1, j2)
                        filename = "%s_maxSSIM.png" % p
                        scipy.misc.imsave(os.path.join(outputFolder, filename), np.nan_to_num(image[0:3,:,:].transpose((2,1,0))))
                        filename = "%s_maxSSIM_gt.png" % p
                        scipy.misc.imsave(os.path.join(outputFolder, filename), np.nan_to_num(gt_image[0:3,:,:].transpose((2,1,0))))
                    if ssimColor < minSSIM[k]:
                        minSSIM[k] = ssimColor
                        minSSIMIndex[k] = (j1, j2)
                        filename = "%s_minSSIM.png" % p
                        scipy.misc.imsave(os.path.join(outputFolder, filename), np.nan_to_num(image[0:3,:,:].transpose((2,1,0))))
                        filename = "%s_minSSIM_gt.png" % p
                        scipy.misc.imsave(os.path.join(outputFolder, filename), np.nan_to_num(gt_image[0:3,:,:].transpose((2,1,0))))

                    #save image
                    #filename = "frame_%d_%d_%s.png" % (j1, j2, p)
                    #scipy.misc.imsave(os.path.join(outputFolder, filename), image[0:3,:,:].transpose((2,1,0)))
                statistics.write("\n")
                statistics.flush()

    except KeyboardInterrupt:
        breakAll = True

    pg.print_progress_bar(SPHERE_SAMPLES * ROTATION_SAMPLES)
    # close statistics of single entries
    statistics.close()
    # write summary statistics
    with open(os.path.join(outputFolder, "AA_Summary_PSNR.txt"), "w") as s:
        s.write("Model\tMin-PSNR-Color\tAt-Index\tMax-PSNR-Color\tAt-Index\tMean-PSNR-Normal\tVar-PSNR-Normal\tMean-PSNR-Color\tVar-PSNR-Color\n")
        for k,m in enumerate(MODELS):
            p = m['name']
            s.write("%s\t%.5f\t(%d:%d)\t%.5f\t(%d:%d)\t%.5f\t%e\t%.5f\t%e\n" % ( 
                    p,
                    minPSNR[k], minPSNRIndex[k][0], minPSNRIndex[k][1],
                    maxPSNR[k], maxPSNRIndex[k][0], maxPSNRIndex[k][1],
                    mvPSNRNormal[k].mean(), mvPSNRNormal[k].var(),
                    mvPSNRColor[k].mean(),  mvPSNRColor[k].var()))
    with open(os.path.join(outputFolder, "AA_Summary_SSIM.txt"), "w") as s:
        s.write("Model\tMin-SSIM-Color\tAt-Index\tMax-SSIM-Color\tAt-Index\tMean-SSIM-Normal\tVar-PSRN-Normal\tMean-SSIM-Color\tVar-SSIM-Color\n")
        for k,m in enumerate(MODELS):
            p = m['name']
            s.write("%s\t%.5f\t(%d:%d)\t%.5f\t(%d:%d)\t%.5f\t%e\t%.5f\t%e\n" % ( 
                    p,
                    minSSIM[k], minSSIMIndex[k][0], minSSIMIndex[k][1],
                    maxSSIM[k], maxSSIMIndex[k][0], maxSSIMIndex[k][1],
                    mvSSIMNormal[k].mean(), mvSSIMNormal[k].var(),
                    mvSSIMColor[k].mean(),  mvSSIMColor[k].var()))

    renderer.close()
    if breakAll:
        break

print('Done')