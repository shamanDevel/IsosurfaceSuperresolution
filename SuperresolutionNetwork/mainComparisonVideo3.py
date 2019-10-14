import math
import os
import os.path
import time
import sys

import numpy as np
import scipy.misc
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F
import skimage

import imageio
from PIL import ImageFont, ImageDraw, Image
from console_progressbar import ProgressBar

import inference
from utils import ScreenSpaceShading

########################################
# Basic comfiguration
########################################
PREVIEW = False
OUTPUT_FIRST_IMAGE = False
CPU_SUPERRES = False
SHOW_DIFFERENCE = False
RENDERER = '../bin/GPURendererDirect.dll'
DATA_DIR_GPU = "C:/Users/ga38cat/Documents/isosurface-super-resolution-data/"
MODEL_DIR = "D:/VolumeSuperResolution"

UPSCALING = 4

OUTPUT_FOLDER = 'D:/VolumeSuperResolution/comparisonVideo3' + ("_diff" if SHOW_DIFFERENCE else "")
FPS = 25
BACKGROUND = [1,1,1]
RESOLUTION = (1920, 1080)
RESOLUTION_LOW = (RESOLUTION[0]//UPSCALING, RESOLUTION[1]//UPSCALING)

########################################
# Material + Camera
########################################

camera = inference.Camera(RESOLUTION[0], RESOLUTION[0], [0,0,-1])
camera.currentDistance = 2.3
camera.currentPitch = 0.38
camera.orientation = inference.Orientation.Yp

class Scene:
    file = None
    isovalue = 0.36
    light = "camera"
    temporalConsistency = False
    depthMin = None
    depthMax = None
    aoSamples = 4 #256 #4
    aoRadius = 0.05
scene = Scene()

cudaDevice = torch.device('cuda')
cpuDevice = torch.device('cpu')

shading = ScreenSpaceShading(cpuDevice if CPU_SUPERRES else cudaDevice)
shading.fov(30)
shading.light_direction(np.array([0.0,0.0,1.0]))
shading.ambient_light_color(np.array([0.01, 0.01, 0.01]))
shading.diffuse_light_color(np.array([0.8, 0.8, 0.8]))
shading.specular_light_color(np.array([0.02, 0.02, 0.02]))
shading.specular_exponent(4)
shading.material_color(np.array([1.0, 1.0, 1.0]))
shading.ambient_occlusion(1.0)
shading.background(np.array(BACKGROUND))

########################################
# HELPER
########################################
def smoothstep(a, b, t):
    x = np.clip((t-a)/(b-a), 0.0, 1.0)
    return x * x * (3 - 2*x)
def smootherstep(a, b, t):
    x = np.clip((t-a)/(b-a), 0.0, 1.0)
    return x * x * x * (x * (x * 6 - 15) + 10);

class BreakRenderingException(Exception):
    pass

########################################
# Scenes
########################################

def Ejecta1a(render_fun):
    scene.file = 'volumes/vbx/snapshot_272_512_ushort.vbx'
    # simple test: rotation
    scene.isovalue = 0.36
    scene.temporalConsistency = False
    scene.depthMin = 0.9390
    scene.depthMax = 0.9477
    camera.currentDistance = 3.6
    camera.currentPitch = 0.38
    camera.currentYaw = 4
    camera.orientation = inference.Orientation.Yp
    NUM_FRAMES = 100
    NUM_ROTATIONS = 1
    for i in range(NUM_FRAMES+1):
        render_fun()
        camera.currentYaw += 2 * math.pi / NUM_FRAMES

def Ejecta1a_v2(render_fun):
    scene.file = 'volumes/vbx/snapshot_272_512_ushort.vbx'
    # simple test: rotation
    scene.isovalue = 0.36
    scene.temporalConsistency = False
    scene.depthMin = 0.85
    scene.depthMax = 0.999
    camera.currentDistance = 3.6
    camera.currentPitch = 0.38
    camera.currentYaw = 4
    camera.orientation = inference.Orientation.Yp
    NUM_FRAMES = 100
    NUM_ROTATIONS = 1
    for i in range(NUM_FRAMES+1):
        render_fun()
        camera.currentYaw += 2 * math.pi / NUM_FRAMES

def Ejecta1b(render_fun):
    scene.file = 'volumes/vbx/snapshot_272_512_ushort.vbx'
    # simple test: rotation
    scene.isovalue = 0.50
    scene.temporalConsistency = False
    scene.depthMin = 0.9390
    scene.depthMax = 0.9477
    camera.currentDistance = 3.6
    camera.currentPitch = 0.38
    camera.currentYaw = 4
    camera.orientation = inference.Orientation.Yp
    NUM_FRAMES = 100
    NUM_ROTATIONS = 1
    for i in range(NUM_FRAMES+1):
        render_fun()
        camera.currentYaw += 2 * math.pi / NUM_FRAMES

def Ejecta2(render_fun):
    scene.file = 'volumes/vbx/snapshot_272_512_ushort.vbx'
    # changing isosurface
    MIN_ISOSURFACE = 0.36
    MAX_ISOSURFACE = 0.50
    scene.temporalConsistency = False
    scene.depthMin = 0.9390
    scene.depthMax = 0.9477
    camera.currentDistance = 3.6
    camera.currentPitch = 0.38
    camera.currentYaw = 4
    camera.orientation = inference.Orientation.Yp
    NUM_FRAMES = 50
    for i in range(NUM_FRAMES+1):
        scene.isovalue = MIN_ISOSURFACE + (MAX_ISOSURFACE-MIN_ISOSURFACE) * smootherstep(0, NUM_FRAMES, i)
        render_fun()

def Ejecta3(render_fun):
    scene.file = 'volumes/vbx/snapshot_272_512_ushort.vbx'
    # changing light direction
    RADIUS = 1
    light_direction = np.array([0.0,0.0,1.0])
    scene.temporalConsistency = False
    scene.depthMin = 0.9390
    scene.depthMax = 0.9477
    camera.currentDistance = 3.6
    camera.currentPitch = 0.38
    camera.currentYaw = 4
    camera.orientation = inference.Orientation.Yp
    NUM_FRAMES = 100
    for i in range(NUM_FRAMES+1):
        phi = i * 2 * math.pi / NUM_FRAMES
        r = (1 - math.cos(phi)) * RADIUS
        light_direction[0] = r * math.cos(phi)
        light_direction[1] = r * math.sin(phi)
        shading.light_direction(light_direction)
        render_fun(True if i==0 else False, True if i==0 else False)
    shading.light_direction(np.array([0.0,0.0,1.0]))

def Ejecta4(render_fun):
    scene.file = 'volumes/vbx/snapshot_272_512_ushort.vbx'
    # changing color
    scene.temporalConsistency = False
    scene.depthMin = 0.9390
    scene.depthMax = 0.9477
    camera.currentDistance = 3.6
    camera.currentPitch = 0.38
    camera.currentYaw = 4
    camera.orientation = inference.Orientation.Yp
    NUM_FRAMES = 100
    for i in range(NUM_FRAMES+1):
        t = i / NUM_FRAMES
        L = 50+85*math.pow(0.5*(1+math.cos(2*math.pi*t)),3)
        A = 100 * math.cos(2*math.pi*t)
        B = 100 * math.sin(2*math.pi*t)
        color = skimage.color.lab2rgb(np.array([[[L, A, B]]], dtype=float))[0,0]
        #print(L,A,B,"->",color)
        shading.material_color(np.array([color[0], color[1], color[2]]))
        render_fun(True if i==0 else False, True if i==0 else False)
    shading.material_color(np.array([1.0, 1.0, 1.0]))

def Ejecta5(render_fun):
    scene.file = 'volumes/vbx/snapshot_272_512_ushort.vbx'
    # changing zoom
    MAX_FOV = shading.get_fov()
    MIN_FOV = 5
    scene.temporalConsistency = True
    scene.depthMin = 0.9390
    scene.depthMax = 0.9477
    camera.currentDistance = 3.6
    camera.currentPitch = 0.38
    camera.currentYaw = 4
    camera.orientation = inference.Orientation.Yp
    NUM_FRAMES = 100
    for i in range(NUM_FRAMES+1):
        fov = MAX_FOV - (MAX_FOV-MIN_FOV) * math.sin(0.5*math.pi*i/NUM_FRAMES)
        shading.fov(fov)
        render_fun()
    shading.fov(MAX_FOV)

def RM1a(render_fun):
    scene.file = 'volumes/vbx/ppmt273_1024_border.vbx'
    # simple test: rotation
    scene.isovalue = 0.34
    scene.temporalConsistency = True
    scene.depthMin = 0.73
    scene.depthMax = 0.93
    camera.currentDistance = 1.12
    camera.currentPitch = 0.522
    camera.currentYaw = 4
    camera.orientation = inference.Orientation.Zm
    NUM_FRAMES = 100
    NUM_ROTATIONS = 1
    for i in range(NUM_FRAMES+1):
        render_fun()
        camera.currentYaw += 2 * math.pi / NUM_FRAMES

def RM1b(render_fun):
    scene.file = 'volumes/vbx/ppmt273_1024_border.vbx'
    # simple test: rotation
    scene.isovalue = 0.34
    scene.temporalConsistency = True
    scene.depthMin = 0.73
    scene.depthMax = 0.93
    camera.currentDistance = 3.0
    camera.currentPitch = 0.522
    camera.currentYaw = 4
    camera.orientation = inference.Orientation.Zm
    NUM_FRAMES = 100
    NUM_ROTATIONS = 1
    for i in range(NUM_FRAMES+1):
        render_fun()
        camera.currentYaw += 2 * math.pi / NUM_FRAMES

def RM2(render_fun):
    scene.file = 'volumes/vbx/ppmt273_1024_border.vbx'
    # zoom
    MIN_DIST = 1.12
    MAX_DIST = 3.0
    scene.isovalue = 0.34
    scene.temporalConsistency = True
    scene.depthMin = 0.73
    scene.depthMax = 0.93
    camera.currentDistance = 3.0
    camera.currentPitch = 0.522
    camera.currentYaw = 4
    camera.orientation = inference.Orientation.Zm
    NUM_FRAMES = 50
    for i in range(NUM_FRAMES+1):
        camera.currentDistance = MIN_DIST + (MAX_DIST-MIN_DIST) * smootherstep(0, NUM_FRAMES, i)
        render_fun()

def Skull1(render_fun):
    scene.file = 'volumes/vbx/vmhead256cubed.vbx'
    # simple test: rotation
    scene.isovalue = 0.31
    scene.temporalConsistency = True
    scene.depthMin = 0.92
    scene.depthMax = 0.94
    camera.currentDistance = 3.0
    camera.currentPitch = 0.0
    camera.currentYaw = math.pi
    camera.orientation = inference.Orientation.Zm
    NUM_FRAMES = 100
    NUM_ROTATIONS = 1
    for i in range(NUM_FRAMES+1):
        render_fun()
        camera.currentYaw += 2 * math.pi / NUM_FRAMES

def Thorax1(render_fun):
    scene.file = 'volumes/vbx/cleveland70.vbx'
    # simple test: rotation
    scene.isovalue = 0.02
    scene.temporalConsistency = True
    scene.depthMin = 0.91
    scene.depthMax = 0.93
    camera.currentDistance = 2.5
    camera.currentPitch = 0.6
    camera.currentYaw = math.pi
    camera.orientation = inference.Orientation.Zm
    NUM_FRAMES = 100
    NUM_ROTATIONS = 1
    for i in range(NUM_FRAMES+1):
        render_fun()
        camera.currentYaw += 2 * math.pi / NUM_FRAMES

# Scene selection
#Scenes = [Ejecta1a, Ejecta1b, Ejecta2]
#Scenes = [Skull1, Thorax1, RM1a, RM1b, RM2]
Scenes = [Ejecta1a_v2]

########################################
# Networks
########################################

MODEL_GROUND_TRUTH = "<gt>"
MODEL_NEAREST = "<input>"
MODEL_BILINEAR = "<bilinear>"
MODEL_BICUBIC = "<bicubic>"
MODELS = [
    {
        'name': 'nearest',
        'path': MODEL_NEAREST,
        'temporal': False,
        'masking': False
    },
    {
        'name': 'bilinear',
        'path': MODEL_BILINEAR,
        'temporal': False,
        'masking': False
    },
    {
        'name': 'bicubic',
        'path': MODEL_BICUBIC,
        'temporal': False,
        'masking': False
    },
    {
        'name': 'GT',
        'path': MODEL_GROUND_TRUTH,
        'temporal': False,
        'masking': False
    },
    {
        'name': 'L1Clouds',
        'path': 'pretrained_unshaded/gen_l1normalDepth_2.pth',
        'temporal': True,
        'masking': True
    },
    {
        'name': 'L1Ejecta',
        'path': 'pretrained_unshaded/gen_l1normal_allEjecta_epoch_100.pth',
        'temporal': True,
        'masking': True
    },
    ]

########################################
# MAIN
########################################

if PREVIEW:
    MODELS = [MODELS[0]]

# open renderer
renderer = inference.DirectRenderer(RENDERER)

CHANNEL_DEPTH = 0
CHANNEL_NORMAL = 1
CHANNEL_AO = 2
CHANNEL_COLOR_NOAO = 3
CHANNEL_COLOR_WITHAO = 4
CHANNEL_NAMES = ["depth", "normal", "ao", "colorNoAO", "colorWithAO"]

# load models
models = [None]*len(MODELS)
for i,m in enumerate(MODELS):
    p = m['path']
    if p.endswith('.pth'):
        models[i] = inference.LoadedModel(os.path.join(MODEL_DIR,p), 
                                          cpuDevice if CPU_SUPERRES else cudaDevice, 
                                          UPSCALING)

# LOOP OVER ALL SCENES
for sceneFun in Scenes:
    sceneName = sceneFun.__name__
    print("====================================")
    print(" Render", sceneName)
    print("====================================")

    # create output folder
    outputFolder = os.path.join(OUTPUT_FOLDER, sceneName)
    print("Write output to", outputFolder)
    os.makedirs(outputFolder, exist_ok = True)

    # open output files
    if OUTPUT_FIRST_IMAGE:
        writers = [[os.path.join(outputFolder, "%s_%s.jpg"%(m['name'],channel))
                for channel in CHANNEL_NAMES] 
                    for m in MODELS]
    else:
        writers = [[imageio.get_writer(os.path.join(outputFolder, "%s_%s.mp4"%(m['name'],channel)), macro_block_size = None)
                    for channel in CHANNEL_NAMES] 
                        for m in MODELS]
    print("Output videos created")

    # no gradients anywhere
    torch.set_grad_enabled(False)

    try:
        # define rendering function
        oldFile = None
        frameIndex = 0
        rendered_low = torch.empty(
            (RESOLUTION_LOW[1], RESOLUTION_LOW[0], 12), 
            dtype=torch.float32, 
            device=cudaDevice)
        rendered_low.share_memory_()
        rendered_high = torch.empty(
            (RESOLUTION[1], RESOLUTION[0], 12), 
            dtype=torch.float32, 
            device=cudaDevice)
        rendered_high.share_memory_()
        previous_frames = [None for i in MODELS]
        global_depth_max = 0.0
        global_depth_min = 1.0
        def render(rerender=True, resuperres=True):
            """
            Main render function
            rerender: if True, the volumes are retraced. If false, the previous images are kept
            resuperres: if True, the superresolution is performed again. If False, the previous result is used
            """
            global oldFile, frameIndex, global_depth_max, global_depth_min, previous_frames
            # check if file was changed
            if oldFile != scene.file:
                oldFile = scene.file
                renderer.load(os.path.join(DATA_DIR_GPU, scene.file))
            # send render parameters
            currentOrigin = camera.getOrigin()
            currentLookAt = camera.getLookAt()
            currentUp = camera.getUp()
            renderer.send_command("cameraOrigin", "%5.3f,%5.3f,%5.3f"%(currentOrigin[0], currentOrigin[1], currentOrigin[2]))
            renderer.send_command("cameraLookAt", "%5.3f,%5.3f,%5.3f"%(currentLookAt[0], currentLookAt[1], currentLookAt[2]))
            renderer.send_command("cameraUp", "%5.3f,%5.3f,%5.3f"%(currentUp[0], currentUp[1], currentUp[2]))
            renderer.send_command("cameraFoV", "%.3f"%shading.get_fov())
            renderer.send_command("isovalue", "%5.3f"%float(scene.isovalue))
            renderer.send_command("aoradius", "%5.3f"%float(scene.aoRadius))
            if PREVIEW:
                renderer.send_command("aosamples", "0")
            else:
                renderer.send_command("aosamples", "%d"%scene.aoSamples)

            if rerender:

                # render low resolution
                renderer.send_command("resolution", "%d,%d"%(RESOLUTION_LOW[0], RESOLUTION_LOW[1]))
                renderer.send_command("viewport", "%d,%d,%d,%d"%(0,0,RESOLUTION_LOW[0], RESOLUTION_LOW[1]))
                renderer.render_direct(rendered_low)

                # render high resolution
                if not PREVIEW:
                    renderer.send_command("resolution", "%d,%d"%(RESOLUTION[0], RESOLUTION[1]))
                    renderer.send_command("viewport", "%d,%d,%d,%d"%(0,0,RESOLUTION[0], RESOLUTION[1]))
                    renderer.render_direct(rendered_high)

            # preprocessing
            def preprocess(input):
                input = input.to(cpuDevice if CPU_SUPERRES else cudaDevice).permute(2,0,1)
                output = torch.unsqueeze(input, 0)
                output = torch.cat((
                    output[:,0:3,:,:],
                    output[:,3:4,:,:]*2-1, #transform mask into -1,+1
                    output[:,4: ,:,:]), dim=1)
                #image_shaded_input = torch.cat((output[:,3:4,:,:], output[:,4:8,:,:], output[:,10:11,:,:]), dim=1)
                #image_shaded = torch.clamp(shading(image_shaded_input), 0, 1)
                #output[:,0:3,:,:] = image_shaded
                return output
            processed_low = preprocess(rendered_low)
            processed_high = preprocess(rendered_high)
            # image now contains all channels:
            # 0:3 - color (shaded)
            # 3:4 - mask in -1,+1
            # 4:7 - normal
            # 7:8 - depth
            # 8:10 - flow
            # 10:11 - AO

            # prepare bounds for depth
            depthForBounds = processed_low[:,7:8,:,:]
            maxDepth = torch.max(depthForBounds)
            minDepth = torch.min(
                depthForBounds + torch.le(depthForBounds, 1e-5).type_as(depthForBounds))
            global_depth_max = max(global_depth_max, maxDepth.item())
            global_depth_min = min(global_depth_min, minDepth.item())
            if scene.depthMin is not None:
                minDepth = scene.depthMin
            if scene.depthMax is not None:
                maxDepth = scene.depthMax

            # mask
            if PREVIEW:
                base_mask = F.interpolate(processed_low, scale_factor=UPSCALING, mode='bilinear')[:,3:4,:,:]
            else:
                base_mask = processed_high[:,3:4,:,:]
            base_mask = (base_mask*0.5+0.5)

            # loop through the models
            for model_idx, model in enumerate(MODELS):
                # perform super-resolution
                if model['path'] == MODEL_NEAREST:
                    image = F.interpolate(processed_low, scale_factor=UPSCALING, mode='nearest')
                elif model['path'] == MODEL_BILINEAR:
                    image = F.interpolate(processed_low, scale_factor=UPSCALING, mode='bilinear')
                elif model['path'] == MODEL_BICUBIC:
                    image = F.interpolate(processed_low, scale_factor=UPSCALING, mode='bicubic')
                elif model['path'] == MODEL_GROUND_TRUTH:
                    image = processed_high
                else:
                    # NETWROK
                    if resuperres:
                        # previous frame
                        if scene.temporalConsistency:
                            previous_frame = previous_frames[model_idx]
                        else:
                            previous_frame = None
                        # apply network
                        imageRaw = models[model_idx].inference(processed_low, previous_frame)
                        # post-process
                        imageRaw = torch.cat([
                            torch.clamp(imageRaw[:,0:1,:,:], -1, +1),
                            ScreenSpaceShading.normalize(imageRaw[:,1:4,:,:], dim=1),
                            torch.clamp(imageRaw[:,4:,:,:], 0, 1)
                            ], dim=1)
                        previous_frames[model_idx] = imageRaw
                    else:
                        imageRaw = previous_frames[model_idx]
                    image = F.interpolate(processed_low, scale_factor=UPSCALING, mode='bilinear')
                    #image[:,0:3,:,:] = shading(imageRaw)
                    image[:,3:8,:,:] = imageRaw[:,0:-1,:,:]
                    image[:,10,:,:] = imageRaw[:,-1,:,:]
                    #masking
                    if model['masking']:
                        image[:,3:4,:,:] = base_mask * 2 - 1
                        #image[:,7:8,:,:] = 0 + base_mask * (image[:,7:8,:,:] - 0)
                        image[:,10:11,:,:] = 1 + base_mask * (image[:,10:11,:,:] - 1)

                # shading
                image_shaded_input = torch.cat((image[:,3:4,:,:], image[:,4:8,:,:], image[:,10:11,:,:]), dim=1)
                image_shaded_withAO = torch.clamp(shading(image_shaded_input), 0, 1)
                ao = shading._ao
                shading.ambient_occlusion(0.0)
                image_shaded_noAO = torch.clamp(shading(image_shaded_input), 0, 1)
                shading.ambient_occlusion(ao)

                # perform channel selection
                for channel_idx in range(len(CHANNEL_NAMES)):
                    if channel_idx == CHANNEL_AO:
                        if SHOW_DIFFERENCE and model['path'] != MODEL_GROUND_TRUTH:
                            image[:,10:11,:,:] = 1 - torch.abs(image[:,10:11,:,:])
                        imageRGB = torch.cat((image[:,10:11,:,:], image[:,10:11,:,:], image[:,10:11,:,:]), dim=1)
                    elif channel_idx == CHANNEL_COLOR_NOAO:
                        imageRGB = image_shaded_noAO
                    elif channel_idx == CHANNEL_COLOR_WITHAO:
                        imageRGB = image_shaded_withAO
                    elif channel_idx == CHANNEL_DEPTH:
                        if SHOW_DIFFERENCE and model['path'] != MODEL_GROUND_TRUTH:
                            depthVal = torch.abs(image[:,7:8,:,:] - processed_high[:,7:8,:,:])# / (2*(maxDepth - minDepth))
                        else:
                            depthVal = (image[:,7:8,:,:] - minDepth) / (maxDepth - minDepth)
                        imageRGB = torch.cat((depthVal, depthVal, depthVal), dim=1)
                        imageRGB = 1 - imageRGB
                        imageRGB[imageRGB < 0.05] = 1.0
                        #imageRGB = BACKGROUND[0] + base_mask * (imageRGB - BACKGROUND[0])
                    elif channel_idx == CHANNEL_NORMAL:
                        if SHOW_DIFFERENCE and model['path'] != MODEL_GROUND_TRUTH:
                            diffVal = F.cosine_similarity(image[:,4:7,:,:], processed_high[:,4:7,:,:], dim=1)*0.5+0.5
                            imageRGB = torch.stack((diffVal, diffVal, diffVal), dim=1)
                            #imageRGB = 1 - torch.abs(image[:,4:7,:,:])
                        else:
                            imageRGB = image[:,4:7,:,:] * 0.5 + 0.5
                        imageRGB = BACKGROUND[0] + base_mask * (imageRGB - BACKGROUND[0])
                    imageRGB = torch.clamp(imageRGB, 0, 1)

                    # copy to numpy and write to video
                    imageRGB_cpu = imageRGB.cpu().numpy()[0].transpose((1,2,0))
                    imageRGB_cpu = np.clip(imageRGB_cpu*255,0,255).astype(np.uint8)
                    if OUTPUT_FIRST_IMAGE:
                        scipy.misc.imsave(writers[model_idx][channel_idx], imageRGB_cpu)
                    else:
                        writers[model_idx][channel_idx].append_data(imageRGB_cpu)
            # done with this frame
            frameIndex += 1
            if frameIndex % 10 == 0:
                print(" %d"%frameIndex)

            if OUTPUT_FIRST_IMAGE:
                raise BreakRenderingException()

        # call scene
        print("Render frames")
        sceneFun(render)

    except BreakRenderingException:
        print("Don't render more images")
    finally:
        print("Close writer")
        if not OUTPUT_FIRST_IMAGE:
            for wx in writers:
                for w in wx:
                    w.close()
        renderer.close()

    print("Done")
    print("global depth min:", global_depth_min, "global depth max:", global_depth_max)