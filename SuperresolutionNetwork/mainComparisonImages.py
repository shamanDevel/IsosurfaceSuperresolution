import math
import os
import os.path
import time

import numpy as np
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F

import imageio
from console_progressbar import ProgressBar

import inference
from utils import ScreenSpaceShading


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
    #    'file':'volumes/vbx/vmhead256cubed.vbx',
    #    'name':'vmhead256cubed',
    #    'iso':0.31,
    #    'material':[165, 184, 186],
    #    'ambient':[25, 25, 25],
    #    'diffuse':[255, 255, 255],
    #    'specular':[50, 50, 50],
    #    'distance':1.8
    #},
    #{
    #    'file':'volumes/vbx/cleveland70.vbx',
    #    'name':'cleveland70',
    #    'iso':0.02,
    #    'material':[200, 180, 255],
    #    'ambient':[0, 90, 15],
    #    'diffuse':[121, 119, 255],
    #    'specular':[50, 50, 50],
    #    'distance':2.1
    #},
    #{
    #    'file':'volumes/vbx/snapshot_272_512_ushort.vbx',
    #    'name':'Ejecta-512',
    #    'iso':0.34,
    #    'material':[135, 207, 236],
    #    'ambient':[25, 25, 25],
    #    'diffuse':[255, 255, 255],
    #    'specular':[50, 50, 50],
    #    'distance':2.17
    #},
    #{
    #    'file':'volumes/vbx/ppmt273_1024.vbx',
    #    'name':'RichtmyerMeshkov_1024',
    #    'iso':0.28,
    #    'material':[238, 189, 137],
    #    'ambient':[25, 25, 25],
    #    'diffuse':[255, 255, 255],
    #    'specular':[50, 50, 50],
    #    'distance':2.17
    #},
    {
        'file':'volumes/vbx/snapshot_272_1024.vbx',
        'name':'Ejecta-1024',
        'iso':1.36,
        'material':[135, 207, 236],
        'ambient':[25, 25, 25],
        'diffuse':[255, 255, 255],
        'specular':[50, 50, 50],
        'distance':2.17
    },
    ]

MODEL_INPUT = "<input>"
MODEL_GROUND_TRUTH = "<gt>"
MODELS = [
    {
        'name': 'Input',
        'path': MODEL_INPUT,
        'temporal': False,
        'masking': False
    },
    {
        'name': 'GroundTruth-NoAO',
        'path': MODEL_GROUND_TRUTH,
        'temporal': False,
        'masking': False,
        'ao': False,
    },
    {
        'name': 'GroundTruth-WithAO',
        'path': MODEL_GROUND_TRUTH,
        'temporal': False,
        'masking': False,
        'ao': True,
    },
    {
        'name': 'gen_l1normal',
        'path': 'pretrained_unshaded/gen_l1normal.pth',
        'temporal': True,
        'masking': True
    },
    #{
    #    'name': 'gen_gan_1',
    #    'path': 'pretrained_unshaded/gen_gan_1.pth',
    #    'temporal': True,
    #    'masking': True
    #},
    #{
    #    'name': 'gen_tgan_1',
    #    'path': 'pretrained_unshaded/gen_gan_2b_nAO_wCol.pth',
    #    'temporal': True,
    #    'masking': True
    #},
    ]
UPSCALING = 4

AO_SAMPLES = 128
AO_RADIUS = 0.05
AO_STRENGTH = 0.8
SPECULAR_EXPONENT = 8

# rendering
OUTPUT_FOLDER = 'comparison3'
CAMERA_ORIGIN = [0, 0.8, -1.4] #will be normalized
FRAMES_PER_ROTATION = 500
PRE_STEPS = 5
TIMING_STEPS = 10
RESOLUTIONS = [
    (1920, 1080, 'FullHD'),
    #(4096, 2160, '4K'),
    #(640,480, 'VGA')
    ]

#computation
MASKING = True

# output path
os.makedirs(OUTPUT_FOLDER, exist_ok = True)
def makePath(resolutionName, inputName, modelName, modeName):
    name = '.'.join([resolutionName, inputName, modelName, modeName])
    return os.path.join(OUTPUT_FOLDER, name + ".png")

# Load models
device = torch.device("cuda")
models = [None]*len(MODELS)
for i,m in enumerate(MODELS):
    p = m['path']
    if p is not MODEL_INPUT and p is not MODEL_GROUND_TRUTH:
        models[i] = inference.LoadedModel(os.path.join(MODEL_DIR,p), device, UPSCALING)

# Save float image
def saveImage(path, image):
    # input: Numpy array of shape 1xHxW or 3xHxW
    C, H, W = image.shape
    if C==1:
        image = np.concatenate((image, image, image), axis=0)
    else:
        assert C == 3
    image = image.transpose((1, 2, 0))
    image = np.clip(image*255, 0, 255).astype(np.uint8)
    imageio.imwrite(path, image)
    print('Saved:', path)

# Prepare timing output
timingFile = open(os.path.join(OUTPUT_FOLDER, 'timings.csv'), 'a')
timingFile.write('Resolution, Input, Model, Rendering-Time (sec), Network-Time (sec)\n')

# Render each dataset
for i in range(len(DATASETS)):
    # create renderer
    camera = inference.Camera(1920, 1080, CAMERA_ORIGIN)
    camera.currentDistance = DATASETS[i]['distance']
    material = inference.Material(DATASETS[i]['iso'])
    renderer_path = RENDERER_CPU if DATASETS[i]['file'].endswith('vdb') else RENDERER_GPU
    data_dir = DATA_DIR_CPU if DATASETS[i]['file'].endswith('vdb') else DATA_DIR_GPU
    renderer = inference.Renderer(renderer_path, os.path.join(data_dir, DATASETS[i]['file']), material, camera)
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

    print('Render', DATASETS[i]['name'])
    cameraYaw = camera.currentYaw
    # for each resolution
    for (w, h, resolutionName) in RESOLUTIONS:
        # for each model
        for k, m in enumerate(MODELS):
            p = m['path']
            if p == MODEL_GROUND_TRUTH:
                renderer.send_command("aosamples=%d\n"% \
                    int(AO_SAMPLES if m['ao'] else 0))
            else:
                renderer.send_command("aosamples=0\n")
            w2 = int(w if p == MODEL_GROUND_TRUTH else w // UPSCALING)
            h2 = int(h if p == MODEL_GROUND_TRUTH else h // UPSCALING)
            renderer.send_command("resolution=%d,%d\n"%(w2, h2))
            previous_image = None
            render_time = 0
            superres_time = 0
            camera.currentYaw = cameraYaw
            # for each timestep
            for j in range(PRE_STEPS + TIMING_STEPS):
                print('Render', DATASETS[i]['name'], resolutionName, m['name'], j, (w2,h2))

                # send camera to renderer
                currentOrigin = camera.getOrigin()
                renderer.send_command("cameraOrigin=%5.3f,%5.3f,%5.3f\n"%(currentOrigin[0], currentOrigin[1], currentOrigin[2]))
                camera.currentYaw += 2 * math.pi / FRAMES_PER_ROTATION
                renderer.send_command("render\n")

                # read image and transform
                image = renderer.read_image(w2, h2)
                if j>=PRE_STEPS: render_time += renderer.get_time()
                image = np.concatenate((
                    image[0:3,:,:],
                    image[3:4,:,:]*2-1, #transform mask into -1,+1
                    image[4: ,:,:]), axis=0)

                #mask_clamped = np.clip(image[3,:,:]*0.5+0.5, 0, 1)
                #if p!=MODEL_GROUND_TRUTH:
                #    mask_clamped = cv.resize(mask_clamped,
                #                             dsize=None,
                #                             fx=UPSCALING, 
                #                             fy=UPSCALING, 
                #                             interpolation=cv.INTER_LINEAR)
                #mask_clamped = mask_clamped[np.newaxis,:,:]

                # shade or super-resolution
                p = m['path']
                if p==MODEL_INPUT:
                    image = cv.resize(image.transpose((2, 1, 0)),
                                      dsize=None,
                                      fx=UPSCALING, 
                                      fy=UPSCALING, 
                                      interpolation=cv.INTER_NEAREST)
                    image = image.transpose((2, 1, 0))
                if p==MODEL_INPUT or p==MODEL_GROUND_TRUTH:
                    image_shaded_input = np.concatenate((image[3:4,:,:], image[4:8,:,:], image[10:11,:,:]), axis=0)
                    image_shaded = torch.clamp(shading(torch.unsqueeze(torch.from_numpy(image_shaded_input),0)), 0, 1).numpy()[0]
                    image[0:3,:,:] = image_shaded
                else:
                    if not m['temporal']: previous_image = None
                    imageInput = np.copy(image)
                    image = cv.resize(image.transpose((2, 1, 0)),
                                      dsize=None,
                                      fx=UPSCALING, 
                                      fy=UPSCALING, 
                                      interpolation=cv.INTER_LINEAR).transpose((2, 1, 0))
                    if models[k].unshaded:
                        # unshaded input
                        torch.cuda.synchronize()
                        t0 = time.time()
                        imageRaw = models[k].inference(imageInput, previous_image)
                        torch.cuda.synchronize()
                        if j>=PRE_STEPS: superres_time += time.time()-t0

                        imageRaw = torch.cat([
                            torch.clamp(imageRaw[:,0:1,:,:], -1, +1),
                            ScreenSpaceShading.normalize(imageRaw[:,1:4,:,:], dim=1),
                            torch.clamp(imageRaw[:,4:,:,:], 0, 1)
                            ], dim=1)
                        previous_image = imageRaw
                        imageRawCpu = imageRaw.cpu()
                        imageRGB = shading(imageRawCpu)[0].numpy().transpose((2, 1, 0))
                        image[3:4,:,:] = imageRawCpu[0,0:1,:,:]
                        image[4:7,:,:] = imageRawCpu[0,1:4,:,:]
                        image[7:8,:,:] = imageRawCpu[0,4:5,:,:]
                        image[10:11,:,:] = imageRawCpu[0,5:6,:,:]
                    else:
                        # shaded input
                        image_shaded_input = np.concatenate((imageInput[3:4,:,:], imageInput[4:8,:,:], imageInput[10:11,:,:]), axis=0)
                        image_shaded = torch.clamp(shading(torch.unsqueeze(torch.from_numpy(image_shaded_input),0)), 0, 1).numpy()[0]
                        image_with_color = np.concatenate((
                            #image[0:3,:,:],
                            np.clip(image_shaded, 0, 1),
                            imageInput[3:4,:,:]*0.5+0.5, #transform mask back to [0,1]
                            imageInput[4:,:,:]), axis=0) 
                        
                        torch.cuda.synchronize()
                        t0 = time.time()
                        imageRaw = models[k].inference(image_with_color, previous_image)
                        torch.cuda.synchronize()
                        if j>=PRE_STEPS: superres_time += time.time()-t0
                        
                        imageRaw = torch.clamp(imageRaw, 0, 1)
                        previous_image = imageRaw
                        imageRGB = imageRaw[0].cpu().numpy().transpose((2, 1, 0))

                    if m['masking']:
                        DILATION = 1
                        KERNEL = np.ones((3,3), np.float32)
                        mask = cv.dilate((imageInput[3,:,:]*0.5+0.5).transpose((1, 0)), KERNEL, iterations = DILATION)
                        mask = cv.resize(mask, dsize=None, fx=UPSCALING, fy=UPSCALING, interpolation=cv.INTER_LINEAR)
                        imageRGB = imageRGB * mask[:,:,np.newaxis]
                        image[4:,:,:] = image[4:,:,:] * mask.transpose((1, 0))[np.newaxis,:,:]
                    image[0:3,:,:] = imageRGB.transpose((2, 1, 0))
                
                # save images
                if j==PRE_STEPS+TIMING_STEPS-1:
                    saveImage(
                        makePath(resolutionName, DATASETS[i]['name'], m['name'], 'color'),
                        image[0:3,:,:])
                    saveImage(
                        makePath(resolutionName, DATASETS[i]['name'], m['name'], 'mask'),
                        image[3:4,:,:]*0.5+0.5)
                    saveImage(
                        makePath(resolutionName, DATASETS[i]['name'], m['name'], 'normal'),
                        image[4:7,:,:]*0.5+0.5)
                    saveImage(
                        makePath(resolutionName, DATASETS[i]['name'], m['name'], 'depth'),
                        image[7:8,:,:])
                    saveImage(
                        makePath(resolutionName, DATASETS[i]['name'], m['name'], 'ao'),
                        image[10:11:,:])

            # write info
            timingFile.write('%s, %s, %s, %7.5f, %7.5f\n' % \
                (resolutionName, DATASETS[i]['name'], m['name'], render_time/TIMING_STEPS, superres_time/TIMING_STEPS))

timingFile.close()