import math
import os
import os.path

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
        'file':'clouds/inputVBX/cloud-049.vbx',
        'name':'Cloud - training data',
        'iso':0.1,
        'material':[255, 76, 0],
        'ambient':[25, 25, 25],
        'diffuse':[255, 255, 255],
        'specular':[50, 50, 50],
        'distance':1.8
    },
    {
        'file':'clouds/inputVBX/den0adv_150.vbx',
        'name':'Smoke plume - training data',
        'iso':0.46,
        'material':[165, 184, 186],
        'ambient':[25, 25, 25],
        'diffuse':[255, 255, 255],
        'specular':[50, 50, 50],
        'distance':1.95
    },
    {
        'file':'volumes/vbx/snapshot_039_256.vbx',
        'name':'Ejecta - test data',
        'iso':0.40,
        'material':[138, 129, 255],
        'ambient':[0, 90, 15],
        'diffuse':[121, 119, 255],
        'specular':[50, 50, 50],
        'distance':0.9
    },
    {
        'file':'volumes/vbx/Bonsai.vbx',
        'name':'Bonsai - test data',
        'iso':0.25,
        'material':[0, 173, 0],
        'ambient':[76, 31, 31],
        'diffuse':[255, 233, 191],
        'specular':[30, 30, 30],
        'distance':2.6
    },
    #{
    #    'file':'volumes/vdb/aneurism256.vdb',
    #    'name':'Aneurism - test data',
    #    'iso':0.40,
    #    'material':[255, 255, 255],
    #    'ambient':[0, 0, 64],
    #    'diffuse':[255, 0, 0],
    #    'specular':[50, 50, 50],
    #    'distance':1.95
    #},
    ]
MODEL_INPUT = "<input>"
MODEL_GROUND_TRUTH = "<gt>"
MODELS = [
    {
        'name': 'Nearest (input)',
        'path': MODEL_INPUT,
        'temporal': False,
        'masking': False
    },
    {
        'name': 'Ground Truth',
        'path': MODEL_GROUND_TRUTH,
        'temporal': False,
        'masking': False
    },
    {
        'name': 'Unshaded Input - Perceptual + temporal L1',
        'path': 'pretrained_unshaded/gen_percNormal_tempL2_1.pth',
        'temporal': True,
        'masking': True
    },
    {
        'name': 'Unshaded Input - Temporal GAN',
        'path': 'pretrained_unshaded/gen_tgan_1.pth',
        'temporal': True,
        'masking': True
    },
    ]
UPSCALING = 4

AO_SAMPLES = 512
AO_RADIUS = 0.05
AO_STRENGTH = 1.0
SPECULAR_EXPONENT = 8

#rendering
OUTPUT_FILE = 'comparison6.mp4'
CAMERA_ORIGIN = [0, 0.8, -1.4] #will be normalized
FPS = 25
FRAMES_PER_ROTATION = 500
ROTATIONS_PER_EXAMPLE = 1
EMPTY_FRAMES = 25

#computation
MASKING = True

#layout
RESOLUTION = (1920, 1080)
BORDER = 5
FONT_FILE = "arial.ttf"
DATASET_FONT_SIZE = 25
MODEL_FONT_SIZE = 20
DatasetFont = ImageFont.truetype(FONT_FILE, DATASET_FONT_SIZE)
ModelFont = ImageFont.truetype(FONT_FILE, MODEL_FONT_SIZE)

TitleHeight = DatasetFont.getsize(DATASETS[0]['name'])[1] + BORDER
ModelHeight = ModelFont.getsize(MODELS[0]['name'])[1]
CanvasWidth = (RESOLUTION[0]-3*BORDER)//2
CanvasHeight = (RESOLUTION[1]-TitleHeight - 4*BORDER) // 2
CanvasOffsets = [
    (BORDER, TitleHeight+ModelHeight + 2*BORDER),
    (CanvasWidth + 2*BORDER, TitleHeight+ModelHeight + 2*BORDER),
    (BORDER, TitleHeight+ModelHeight + CanvasHeight + 3*BORDER),
    (CanvasWidth + 2*BORDER, TitleHeight+ModelHeight + CanvasHeight + 3*BORDER),
    ]
CanvasHeight -= ModelHeight
CanvasWidthSmall = CanvasWidth // UPSCALING
CanvasHeightSmall = CanvasHeight // UPSCALING
CanvasWidth = CanvasWidthSmall * UPSCALING
CanvasHeight = CanvasHeightSmall * UPSCALING

# background image and layout
def createBackgroundImage(dataset):
    img = Image.new('RGB', RESOLUTION, color=(0,0,0))
    d = ImageDraw.Draw(img)
    #title
    size = d.textsize(dataset, font=DatasetFont)
    d.text(((RESOLUTION[0]-size[0])//2, BORDER), dataset, font=DatasetFont, fill=(255,255,255))
    #models
    for i in range(4):
        size = d.textsize(MODELS[i]['name'], font=ModelFont)
        d.text((CanvasOffsets[i][0]+(CanvasWidth-size[0])//2, CanvasOffsets[i][1]-ModelHeight), MODELS[i]['name'],
               font=ModelFont, fill=(200,200,200))
    return np.asarray(img)

# create writer
writer = imageio.get_writer(OUTPUT_FILE, fps=FPS)

# Load models
device = torch.device("cuda")
models = [None]*len(MODELS)
for i,m in enumerate(MODELS):
    p = m['path']
    if p is not MODEL_INPUT and p is not MODEL_GROUND_TRUTH:
        models[i] = inference.LoadedModel(os.path.join(MODEL_DIR,p), device, UPSCALING)

# Render each dataset
for i in range(len(DATASETS)):
    if i>0:
        #write empty frames
        empty = np.zeros((RESOLUTION[1], RESOLUTION[0], 3), dtype=np.uint8)
        for j in range(EMPTY_FRAMES):
            writer.append_data(empty)

    background = createBackgroundImage(DATASETS[i]['name'])
    # create renderer
    camera = inference.Camera(CanvasWidth, CanvasHeight, CAMERA_ORIGIN)
    camera.currentDistance = DATASETS[i]['distance']
    material = inference.Material(DATASETS[i]['iso'])
    renderer_path = RENDERER_CPU if DATASETS[i]['file'].endswith('vdb') else RENDERER_GPU
    data_dir = DATA_DIR_CPU if DATASETS[i]['file'].endswith('vdb') else DATA_DIR_GPU
    datasetfile = os.path.join(data_dir, DATASETS[i]['file'])
    print('Open', datasetfile)
    renderer = inference.Renderer(renderer_path, datasetfile, material, camera)
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
    # draw frames
    print('Render', DATASETS[i]['file'])
    pg = ProgressBar(FRAMES_PER_ROTATION * ROTATIONS_PER_EXAMPLE, 'Dataset %d'%(i+1), length=50)
    previous_images = [None]*len(MODELS)
    for j in range(FRAMES_PER_ROTATION * ROTATIONS_PER_EXAMPLE):
        pg.print_progress_bar(j)
        img = np.copy(background)
        #send camera to renderer
        currentOrigin = camera.getOrigin()
        renderer.send_command("cameraOrigin=%5.3f,%5.3f,%5.3f\n"%(currentOrigin[0], currentOrigin[1], currentOrigin[2]))
        #render models
        for k,m in enumerate(MODELS):
            #render
            shading.inverse_ao = False
            p = m['path']
            if p==MODEL_INPUT:
                renderer.send_command("resolution=%d,%d\n"%(CanvasWidthSmall, CanvasHeightSmall))
                renderer.send_command("aosamples=0\n")
                renderer.send_command("render\n")
                image = renderer.read_image(CanvasWidthSmall, CanvasHeightSmall)
                image = np.concatenate((
                    image[0:3,:,:],
                    image[3:4,:,:]*2-1, #transform mask into -1,+1
                    image[4: ,:,:]), axis=0)
                image_shaded_input = np.concatenate((image[3:4,:,:], image[4:8,:,:], image[10:11,:,:]), axis=0)
                image_shaded = torch.clamp(shading(torch.unsqueeze(torch.from_numpy(image_shaded_input),0)), 0, 1).numpy()[0]
                imageRGB = image_shaded[0:3,:,:].transpose((2, 1, 0))
                imageRGB = cv.resize(imageRGB, 
                                     dsize=None, 
                                     fx=UPSCALING, 
                                     fy=UPSCALING, 
                                     interpolation=cv.INTER_NEAREST)
            elif p==MODEL_GROUND_TRUTH:
                renderer.send_command("resolution=%d,%d\n"%(CanvasWidth, CanvasHeight))
                renderer.send_command("aosamples=%d\n"%AO_SAMPLES)
                renderer.send_command("render\n")
                image = renderer.read_image(CanvasWidth, CanvasHeight)
                image = np.concatenate((
                    image[0:3,:,:],
                    image[3:4,:,:]*2-1, #transform mask into -1,+1
                    image[4: ,:,:]), axis=0)
                image_shaded_input = np.concatenate((image[3:4,:,:], image[4:8,:,:], image[10:11,:,:]), axis=0)
                image_shaded = torch.clamp(shading(torch.unsqueeze(torch.from_numpy(image_shaded_input),0)), 0, 1).numpy()[0]
                imageRGB = image_shaded[0:3,:,:].transpose((2, 1, 0))
            else:
                renderer.send_command("resolution=%d,%d\n"%(CanvasWidthSmall, CanvasHeightSmall))
                renderer.send_command("aosamples=0\n")
                renderer.send_command("render\n")
                shading.inverse_ao = models[k].inverse_ao
                
                image = renderer.read_image(CanvasWidthSmall, CanvasHeightSmall)
                image = np.concatenate((
                    image[0:3,:,:],
                    image[3:4,:,:]*2-1, #transform mask into -1,+1
                    image[4: ,:,:]), axis=0)
                image_shaded_input = np.concatenate((image[3:4,:,:], image[4:8,:,:], image[10:11,:,:]), axis=0)
                image_shaded = torch.clamp(shading(torch.unsqueeze(torch.from_numpy(image_shaded_input),0)), 0, 1).numpy()[0]

                previous_image = previous_images[k]
                if not m['temporal']:
                    previous_image = None

                if models[k].unshaded:
                    # unshaded input
                    imageRaw = models[k].inference(image, previous_image)
                    imageRaw = torch.cat([
                        torch.clamp(imageRaw[:,0:1,:,:], -1, +1),
                        ScreenSpaceShading.normalize(imageRaw[:,1:4,:,:], dim=1),
                        torch.clamp(imageRaw[:,4:,:,:], 0, 1)
                        ], dim=1)
                    previous_images[k] = imageRaw
                    imageRGB = shading(imageRaw.cpu())[0].numpy().transpose((2, 1, 0))
                else:
                    # shaded input
                    image_with_color = np.concatenate((
                        #image[0:3,:,:],
                        np.clip(image_shaded, 0, 1),
                        image[3:4,:,:]*0.5+0.5, #transform mask back to [0,1]
                        image[4:,:,:]), axis=0) 
                    imageRaw = models[k].inference(image_with_color, previous_image)
                    imageRaw = torch.clamp(imageRaw, 0, 1)
                    previous_images[k] = imageRaw
                    imageRGB = imageRaw[0].cpu().numpy().transpose((2, 1, 0))

                if m['masking']:
                    DILATION = 1
                    KERNEL = np.ones((3,3), np.float32)
                    mask = cv.dilate((image[3,:,:]*0.5+0.5).transpose((1, 0)), KERNEL, iterations = DILATION)
                    mask = cv.resize(mask, dsize=None, fx=UPSCALING, fy=UPSCALING, interpolation=cv.INTER_LINEAR)
                    imageRGB = imageRGB * mask[:,:,np.newaxis]

            #write into image
            img[CanvasOffsets[k][1]:CanvasOffsets[k][1]+CanvasHeight,
                CanvasOffsets[k][0]:CanvasOffsets[k][0]+CanvasWidth
                :] = np.clip(imageRGB*255, 0, 255).transpose((1, 0, 2)).astype(np.uint8)
        #rotate camera
        camera.currentYaw += 2 * math.pi / FRAMES_PER_ROTATION
        #send to writer
        writer.append_data(img)
    pg.print_progress_bar(FRAMES_PER_ROTATION * ROTATIONS_PER_EXAMPLE)
    # close renderer
    renderer.close()

# done
writer.close()
print("Done")