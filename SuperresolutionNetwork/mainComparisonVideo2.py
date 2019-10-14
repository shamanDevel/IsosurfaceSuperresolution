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
from PIL import ImageFont, ImageDraw, Image
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
    {
        'file':'volumes/vbx/snapshot_272_512_ushort.vbx',
        'name':'Ejecta-512',
        'iso':0.34,
        'material':[255, 255, 255],
        'ambient':[25, 25, 25],
        'diffuse':[172, 177, 179],
        'specular':[35, 35, 35],
        'distance':2.3,
        'orientation': inference.Orientation.Yp,
        'pitch': 0.38,
        'lookAt': [0,0,0]
    },
    {
        'file':'volumes/vbx/ppmt273_1024.vbx',
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
    {
        'name': 'gen_gan_1',
        'path': 'pretrained_unshaded/gen_gan_1.pth',
        'temporal': True,
        'masking': True
    },
    {
        'name': 'gen_tgan_1',
        'path': 'pretrained_unshaded/gen_gan_2b_nAO_wCol.pth',
        'temporal': True,
        'masking': True
    },
    ]
UPSCALING = 4

AO_SAMPLES = 128
AO_RADIUS = 0.05
AO_STRENGTH = 0.8
SPECULAR_EXPONENT = 8

#rendering
OUTPUT_FOLDER = 'comparisonVideo1'
CAMERA_ORIGIN = [0, 0.8, -1.4] #will be normalized
FPS = 25
FRAMES_PER_ROTATION = 100
ROTATIONS_PER_EXAMPLE = 2
FRAME_DUPLICATION = 6
BACKGROUND = [1,1,1]
RESOLUTION = (1920, 1080)

# Load models
device = torch.device("cuda")
models = [None]*len(MODELS)
for i,m in enumerate(MODELS):
    p = m['path']
    if p is not MODEL_INPUT and p is not MODEL_GROUND_TRUTH:
        models[i] = inference.LoadedModel(os.path.join(MODEL_DIR,p), device, UPSCALING)

# create output folder
os.makedirs(OUTPUT_FOLDER, exist_ok = True)

# Render each dataset
for i in range(len(DATASETS)):
    # create renderer
    camera = inference.Camera(RESOLUTION[0], RESOLUTION[0], CAMERA_ORIGIN)
    camera.currentDistance = DATASETS[i]['distance']
    camera.currentPitch = DATASETS[i]['pitch']
    camera.orientation = DATASETS[i]['orientation']
    material = inference.Material(DATASETS[i]['iso'])
    renderer_path = RENDERER_CPU if DATASETS[i]['file'].endswith('vdb') else RENDERER_GPU
    data_dir = DATA_DIR_CPU if DATASETS[i]['file'].endswith('vdb') else DATA_DIR_GPU
    datasetfile = os.path.join(data_dir, DATASETS[i]['file'])
    print('Open', datasetfile)
    renderer = inference.Renderer(renderer_path, datasetfile, material, camera)
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

    # render each model
    for k,m in enumerate(MODELS):
        print('Render', m['name'])
        p = m['path']
        outputName = os.path.join(OUTPUT_FOLDER, "%s.%s.mp4"%(DATASETS[i]['name'], m['name']))
        writer = imageio.get_writer(outputName, fps=FPS)

        camera.currentYaw = 0
        previous_image = None

        if p == MODEL_GROUND_TRUTH:
            renderer.send_command("aosamples=%d\n"% \
                int(AO_SAMPLES if m['ao'] else 0))
        else:
            renderer.send_command("aosamples=0\n")
        w2 = int(RESOLUTION[0] if p == MODEL_GROUND_TRUTH else RESOLUTION[0] // UPSCALING)
        h2 = int(RESOLUTION[1] if p == MODEL_GROUND_TRUTH else RESOLUTION[1] // UPSCALING)
        renderer.send_command("resolution=%d,%d\n"%(w2, h2))

        pg = ProgressBar(FRAMES_PER_ROTATION * ROTATIONS_PER_EXAMPLE, 'Render', length=50)
        for j in range(FRAMES_PER_ROTATION * ROTATIONS_PER_EXAMPLE):
            pg.print_progress_bar(j)
            # send camera
            currentOrigin = np.array(camera.getOrigin()) + np.array(DATASETS[i]['lookAt'])
            currentLookAt = np.array(camera.getLookAt()) + np.array(DATASETS[i]['lookAt'])
            currentUp = camera.getUp()
            renderer.send_command("cameraOrigin=%5.3f,%5.3f,%5.3f\n"%(currentOrigin[0], currentOrigin[1], currentOrigin[2]))
            renderer.send_command("cameraLookAt=%5.3f,%5.3f,%5.3f\n"%(currentLookAt[0], currentLookAt[1], currentLookAt[2]))
            renderer.send_command("cameraUp=%5.3f,%5.3f,%5.3f\n"%(currentUp[0], currentUp[1], currentUp[2]))
            camera.currentYaw += 2 * math.pi / FRAMES_PER_ROTATION
            # render and read back
            renderer.send_command("render\n")
            image = renderer.read_image(w2, h2)
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
            if p==MODEL_INPUT:
                image = cv.resize(image.transpose((2, 1, 0)),
                                    dsize=None,
                                    fx=UPSCALING, 
                                    fy=UPSCALING, 
                                    interpolation=cv.INTER_NEAREST)
                image = image.transpose((2, 1, 0))
            elif p==MODEL_GROUND_TRUTH:
                pass # nothing to do
            else: # network
                if not m['temporal']: previous_image = None
                imageInput = np.copy(image)
                #image = cv.resize(image.transpose((2, 1, 0)),
                #                    dsize=None,
                #                    fx=UPSCALING, 
                #                    fy=UPSCALING, 
                #                    interpolation=cv.INTER_LINEAR).transpose((2, 1, 0))
                if models[k].unshaded:
                    # unshaded input
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
                    image[0:3,:,:] = shading(imageRawCpu)[0].numpy()
                    imageRawCpu = imageRawCpu[0].numpy()
                    image[[3,4,5,6,7,10],:,:] = imageRawCpu

                else:
                    # shaded input
                    image_shaded_input = np.concatenate((imageInput[3:4,:,:], imageInput[4:8,:,:], imageInput[10:11,:,:]), axis=0)
                    image_shaded = torch.clamp(shading(torch.unsqueeze(torch.from_numpy(image_shaded_input),0)), 0, 1).numpy()[0]
                    image_with_color = np.concatenate((
                        #image[0:3,:,:],
                        np.clip(image_shaded, 0, 1),
                        imageInput[3:4,:,:]*0.5+0.5, #transform mask back to [0,1]
                        imageInput[4:,:,:]), axis=0) 
                        
                    imageRaw = models[k].inference(image_with_color, previous_image)
                        
                    imageRaw = torch.clamp(imageRaw, 0, 1)
                    previous_image = imageRaw
                    
                    image = image.transpose((2, 1, 0))
                    image = cv.resize(image, dsize=None, fx=UPSCALING, fy=UPSCALING, interpolation=cv.INTER_LINEAR)
                    image = image.transpose((2, 1, 0))
                    base_mask = np.copy(image[3,:,:])
                    imageRawCpu = imageRaw[0].cpu().numpy()
                    image[0:3,:,:] = imageRawCpu[0:3,:,:]

                if m['masking']:
                    #DILATION = 1
                    #KERNEL = np.ones((3,3), np.float32)
                    #mask = cv.dilate((imageInput[3,:,:]*0.5+0.5).transpose((1, 0)), KERNEL, iterations = DILATION)
                    #mask = cv.resize(mask, dsize=None, fx=UPSCALING, fy=UPSCALING, interpolation=cv.INTER_LINEAR)
                    #imageRGB = imageRGB * mask[:,:,np.newaxis]
                    #image[4:,:,:] = image[4:,:,:] * mask.transpose((1, 0))[np.newaxis,:,:]
                    mask = (base_mask*0.5+0.5)
                    image = BACKGROUND[0] + mask[np.newaxis,:,:] * (image - BACKGROUND[0])

            # select color image and add to video
            imageRGB = image[0:3,:,:].transpose((1,2,0))
            imageRGB = np.clip(imageRGB*255,0,255).astype(np.uint8)
            for d in range(FRAME_DUPLICATION):
                writer.append_data(imageRGB)

        pg.print_progress_bar(FRAMES_PER_ROTATION * ROTATIONS_PER_EXAMPLE)
        writer.close()
    renderer.close()

print('Done')