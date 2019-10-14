"""
Interactive GUI
"""

import argparse
import math
import os
import os.path
import subprocess
import time

import numpy as np
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F

import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import PIL as pl
from PIL import ImageTk, Image
from tkcolorpicker import askcolor

import models
import inference
from utils import ScreenSpaceShading

##profiling
#import line_profiler
#import atexit
#profile = line_profiler.LineProfiler()
#atexit.register(profile.print_stats)

########################################
# CONFIGURATION
########################################

DEFAULT_MODEL_DIR = ""

DEFUALT_DATA_DIR = ""

INPUT_FILETYPES = (("GVDB file (GPU)", "*.vbx"), ("OpenVDB file (CPU)", "*.vdb"))

DEFAULT_RESOLUTION = (1280, 960)
#DEFAULT_RESOLUTION = (800, 600)

parser = argparse.ArgumentParser(description='Interactive video super resolution')

parser.add_argument('--model', type=str, default=None, help="""
The checkpoint containing the superresolution network.
If you want to load multiple models, use a comma to seperate those
""")
parser.add_argument('--rendererCPU', type=str, default='../bin/CPURenderer.exe',
                    help="The path to CPURenderer.exe")
parser.add_argument('--rendererGPU', type=str, default='../bin/GPURenderer.exe',
                    help="The path to GPURenderer.exe")
parser.add_argument('--rendererDirectGPU', type=str, default='../bin/GPURendererDirect.dll',
                    help="The path to GPURendererDirect.dll")
parser.add_argument('--resX', type=int, default=DEFAULT_RESOLUTION[0]//4, help="X-resolution")
parser.add_argument('--resY', type=int, default=DEFAULT_RESOLUTION[1]//4, help="Y-resolution")
parser.add_argument('--input', type=str, default=None,#"../../data/clouds/input/cloud-049.vdb",
                    help="The volume file to visualize")
parser.add_argument("--iso", type=str, default=0.1, help="isovalue to render")

opt = parser.parse_args()
if not torch.cuda.is_available():
    raise Exception("No GPU found!")
device = torch.device("cuda")

upscale_factor = 4

# Specifies how the data is passed from the renderer to the GUI/Network
# False: over the CPU / stream piping
# True: over the GPU via a shared tensor
RENDERER_DIRECT_WRITE = True

########################################
# GUI
########################################

class GUI(tk.Tk):

    MODE_NEAREST = 1
    MODE_BILINEAR = 2
    MODE_BICUBIC = 3
    #MODE_NORMAL = 4
    #MODE_FLOW = 5
    #MODE_PREVIOUS_WARPED = 6
    MODE_GROUND_TRUTH = 4#7
    MODE_SUPERRESOLUTION = 5#8

    CHANNEL_MASK = 1
    CHANNEL_NORMAL = 2
    CHANNEL_DEPTH = 3
    CHANNEL_AO = 4
    CHANNEL_FLOW = 5
    CHANNEL_COLOR = 6

    BACKGROUND_BLACK = 0
    BACKGROUND_WHITE = 1

    def __init__(self):
        tk.Tk.__init__(self)
        self.title('Isosurface-Superresolution')
        self.input_name = None

        # members to be filled later
        self.pilImage = None
        self.tikImage = None
        self.models_ = None
        self.renderer_ = None
        self.previous_rgb_images = None

        #material
        self.material = inference.Material(opt.iso)

        #camera
        self.camera = inference.Camera(opt.resX, opt.resY)

        # root
        self.root_panel = tk.Frame(self)
        self.root_panel.pack(side="bottom", fill="both", expand="yes")
        self.panel = tk.Label(self.root_panel)
        self.setImage(np.zeros((opt.resX * upscale_factor, opt.resY * upscale_factor, 3)))
        self.panel.pack(side="left", fill="both", expand="yes")

        options1 = tk.Label(self.root_panel)
        options2 = tk.Label(self.root_panel)
        # Input
        inputFrame = ttk.LabelFrame(options1, text="Input", relief=tk.RIDGE)
        tk.Button(inputFrame, text="Load Input", command=lambda : self.initRenderer(None, True)).pack()
        tk.Button(inputFrame, text="Load Networks", command=lambda : self.initNetwork(None, True)).pack()
        inputFrame.pack(fill=tk.X)
        # Camera
        self.initCamera(options1)
        # Output settings
        self.initOutput(options1)
        # Render settings
        self.initRenderSettings(options1)
        # Superresolution mode
        self.channelsFrame = ttk.LabelFrame(options2, text="Channels", relief=tk.RIDGE)
        self.superresolutionFrame = ttk.LabelFrame(options2, text="Superresolution", relief=tk.RIDGE)
        self.superresolutionOptions = ttk.LabelFrame(options2, text="Options", relief=tk.RIDGE)
        self.initChannels()
        self.initSuperresOptions()

        options1.pack(side="left")
        options2.pack(side="left")

        #frame counter
        self.fps = list()
        self.render_time = 0

        self.previous_frame = np.zeros((3, opt.resY * upscale_factor, opt.resX * upscale_factor))
        self.upres_frame = np.zeros((3, opt.resY * upscale_factor, opt.resX * upscale_factor))
        self.last_renderer_dir = None
        self.last_network_dir = None

    def initCamera(self, options):
        # base orientation
        cameraFrame = ttk.LabelFrame(options, text="Camera", relief=tk.RIDGE)

        def updateOrientation():
            self.camera.orientation = inference.Orientation(self.camera_orientation.get())
            self.updateImage(only_foc_changed=False, clear_fps=False)
        self.camera_orientation = tk.IntVar()
        self.camera_orientation.set(inference.Orientation.Yp.value)
        self.camera_orientation.trace_add('write', lambda a,b,c : updateOrientation())
        for name, member in inference.Orientation.__members__.items():
            tk.Radiobutton(cameraFrame, 
                           text=member.name, 
                           variable=self.camera_orientation, 
                           value=member.value).pack(anchor=tk.W)

        self.camera_fov = 45
        def setCameraFov(e):
            self.camera_fov = float(e)
            self.camera_fov_slider.config(label='Field of View: %.1f°'%float(e))
            self.updateImage(only_foc_changed=False, clear_fps=False)
        self.camera_fov_slider = tk.Scale(
            cameraFrame,
            from_=0.1, to=90,
            orient=tk.HORIZONTAL,
            resolution=0.1,
            label="Field of View: %.1f°" % self.camera_fov,
            showvalue=0,
            command=setCameraFov)
        self.camera_fov_slider.set(self.camera_fov)
        self.camera_fov_slider.pack(anchor=tk.W, fill=tk.X)

        # focus of context
        self.foc_center = (0,0)

        self.enable_foc = tk.IntVar()
        self.enable_foc.set(0)
        tk.Checkbutton(cameraFrame, 
                       text="Enable Focus-of-Context", 
                       variable=self.enable_foc).pack(anchor=tk.W)
        self.enable_foc.trace_add('write', 
             lambda a,b,c : self.updateImage(only_foc_changed=True, clear_fps=False))

        self.foc_window_size = 20
        def setFocWindowSize(e):
            self.foc_window_size = int(e)
            self.foc_window_size_slider.config(label='FoC Window Size: %d'%int(e))
            self.updateImage(only_foc_changed=True, clear_fps=False)
        self.foc_window_size_slider = tk.Scale(
            cameraFrame,
            from_=10, to=200,
            orient=tk.HORIZONTAL,
            resolution=1,
            label="FoC Window Size: %d" % self.foc_window_size,
            showvalue=0,
            command=setFocWindowSize)
        self.foc_window_size_slider.set(self.foc_window_size)
        self.foc_window_size_slider.pack(anchor=tk.W, fill=tk.X)

        self.foc_blur_radius = 5
        def setFocBlurRadius(e):
            self.foc_blur_radius = int(e)
            self.foc_blur_radius_slider.config(label='FoC Blur Radius: %d'%int(e))
            self.updateImage(only_foc_changed=True, clear_fps=False)
        self.foc_blur_radius_slider = tk.Scale(
            cameraFrame,
            from_=1, to=50,
            orient=tk.HORIZONTAL,
            resolution=1,
            label="FoC Blur Radius: %d" % self.foc_blur_radius,
            showvalue=0,
            command=setFocBlurRadius)
        self.foc_blur_radius_slider.set(self.foc_blur_radius)
        self.foc_blur_radius_slider.pack(anchor=tk.W, fill=tk.X)

        cameraFrame.pack(fill=tk.X)

        # camera movement
        self.panel.bind("<ButtonPress-1>", self.StartMove)
        self.panel.bind("<ButtonRelease-1>", self.StopMove)
        self.panel.bind("<Motion>", self.OnMouseMotion)
        self.panel.bind("<B1-Motion>", self.OnMouseDrag)
        self.panel.bind("<MouseWheel>", self.OnMouseWheel)

    def initOutput(self, options):
        outputFrame = ttk.LabelFrame(options, text="Render Settings", relief=tk.RIDGE)

        tk.Label(outputFrame, text='Background').pack(anchor=tk.W)
        self.background_mode = tk.IntVar()
        self.background_mode.set(GUI.BACKGROUND_BLACK)
        self.background_mode.trace_add('write', lambda a,b,c : self.updateImage(only_foc_changed=False, clear_fps=False))
        tk.Radiobutton(outputFrame, text="Black", variable=self.background_mode, value=GUI.BACKGROUND_BLACK).pack(anchor=tk.W)
        tk.Radiobutton(outputFrame, text="White", variable=self.background_mode, value=GUI.BACKGROUND_WHITE).pack(anchor=tk.W)

        tk.Button(outputFrame, text="Screenshot", command=lambda : self.saveScreenshot()).pack()

        outputFrame.pack(fill=tk.X)

    def initRenderSettings(self, options):
        renderFrame = ttk.LabelFrame(options, text="Render Settings", relief=tk.RIDGE)

        # Isovalue
        def setIsovalue(e):
            self.isoslider.config(label='Isovalue: %5.3f'%float(e))
            self.updateImage(only_foc_changed=False, clear_fps=False)
        self.isoslider = tk.Scale(renderFrame, 
                                  from_=0.01, to=3.0, 
                                  orient=tk.HORIZONTAL, 
                                  resolution=0.01,
                                  label='Isovalue: %4.2f'%self.material.isovalue,
                                  showvalue=0,
                                  command=setIsovalue)
        self.isoslider.set(self.material.isovalue)
        self.isoslider.pack(anchor=tk.W, fill=tk.X)

        # Shading
        self.shading = ScreenSpaceShading(device)
        self.shading.fov(30)
        self.ambient_light_color = np.array([0.1,0.1,0.1])
        self.shading.ambient_light_color(self.ambient_light_color)
        self.diffuse_light_color = np.array([1.0, 1.0, 1.0])
        self.shading.diffuse_light_color(self.diffuse_light_color)
        self.specular_light_color = np.array([0.2, 0.2, 0.2])
        self.shading.specular_light_color(self.specular_light_color)
        self.shading.specular_exponent(16)
        self.shading.light_direction(np.array([0.1,0.1,1.0]))
        self.material_color = np.array([1.0,1.0,1.0])#[1.0, 0.3, 0.0])
        self.shading.material_color(self.material_color)
        self.shading.ambient_occlusion(0.5)

        def toTuple(npArray):
            return (int(npArray[0]*255), int(npArray[1]*255), int(npArray[2]*255))
        def fromTuple(t):
            return np.array([t[0]/255.0, t[1]/255.0, t[2]/255.0])
        def toString(npArray):
            t = toTuple(npArray)
            return "#%02x%02x%02x"%(t[0],t[1],t[2])

        def getMaterialColor():
            color,_ = askcolor(parent=self, title="Material Color", color=toTuple(self.material_color))
            if color is not None:
                self.material_color = fromTuple(color)
                self.materialColorButton.configure(text = 'Material Color: '+toString(self.material_color))
                self.shading.material_color(self.material_color)
                self.updateImage(only_foc_changed=False, clear_fps=False)
                print("new ambient light: ", self.ambient_light_color)
        self.materialColorButton = tk.Button(renderFrame, 
                                         text='Material Color: '+toString(self.material_color), 
                                         command=getMaterialColor)
        self.materialColorButton.pack()

        def getAmbientColor():
            color,_ = askcolor(parent=self, title="Ambient Color", color=toTuple(self.ambient_light_color))
            if color is not None:
                self.ambient_light_color = fromTuple(color)
                self.ambientLightButton.configure(text = 'Ambient Color: '+toString(self.ambient_light_color))
                self.shading.ambient_light_color(self.ambient_light_color)
                self.updateImage(only_foc_changed=False, clear_fps=False)
                print("new ambient light: ", self.ambient_light_color)
        self.ambientLightButton = tk.Button(renderFrame, 
                                         text='Ambient Color: '+toString(self.ambient_light_color), 
                                         command=getAmbientColor)
        self.ambientLightButton.pack()

        def getDiffuseColor():
            color,_ = askcolor(parent=self, title="Diffuse Color", color=toTuple(self.diffuse_light_color))
            if color is not None:
                self.diffuse_light_color = fromTuple(color)
                self.diffuseLightButton.configure(text = 'Diffuse Color: '+toString(self.diffuse_light_color))
                self.shading.diffuse_light_color(self.diffuse_light_color)
                self.updateImage(only_foc_changed=False, clear_fps=False)
                print("new diffuse light: ", self.diffuse_light_color)
        self.diffuseLightButton = tk.Button(renderFrame, 
                                         text='Diffuse Color: '+toString(self.diffuse_light_color), 
                                         command=getDiffuseColor)
        self.diffuseLightButton.pack()

        def getSpecularColor():
            color,_ = askcolor(parent=self, title="Specular Color", color=toTuple(self.specular_light_color))
            if color is not None:
                self.specular_light_color = fromTuple(color)
                self.specularLightButton.configure(text = 'Specular Color: '+toString(self.specular_light_color))
                self.shading.specular_light_color(self.specular_light_color)
                self.updateImage(only_foc_changed=False, clear_fps=False)
                print("new specular light: ", self.specular_light_color)
        self.specularLightButton = tk.Button(renderFrame, 
                                         text='Specular Color: '+toString(self.specular_light_color), 
                                         command=getSpecularColor)
        self.specularLightButton.pack()

        def updateSpecularExponent(e):
            self.shading.specular_exponent(int(e))
            self.specular_exponent.config(label='Specular exponent: %d'%int(self.shading._specular_exponent))
            self.updateImage(only_foc_changed=False, clear_fps=False)
        self.specular_exponent = tk.Scale(renderFrame, 
                                  from_=8, to=64, 
                                  orient=tk.HORIZONTAL, 
                                  resolution=1,
                                  showvalue=0,
                                  label='Specular exponent: %d'%int(self.shading._specular_exponent),
                                  command=updateSpecularExponent)
        self.specular_exponent.set(self.shading._specular_exponent)
        self.specular_exponent.pack(anchor=tk.W, fill=tk.X)

        # AMBIENT OCCLUSION
        def setAoSamples(e):
            self.aoSamplesSlider.config(label='AO Samples: %d'%int(e))
            self.updateImage(only_foc_changed=False, clear_fps=False)
        self.aoSamplesSlider = tk.Scale(renderFrame, 
                                  from_=0, to=512, 
                                  orient=tk.HORIZONTAL, 
                                  resolution=1,
                                  label='AO Samples: %d'%int(16),
                                  showvalue=0,
                                  command=setAoSamples)
        self.aoSamplesSlider.set(16)
        self.aoSamplesSlider.pack(anchor=tk.W, fill=tk.X)

        def setAoRadius(e):
            self.aoRadiusSlider.config(label='AO Radius: %5.3f'%float(e))
            self.updateImage(only_foc_changed=False, clear_fps=False)
        self.aoRadiusSlider = tk.Scale(renderFrame, 
                                  from_=0.001, to=0.1, 
                                  orient=tk.HORIZONTAL, 
                                  resolution=0.001,
                                  label='AO Samples: %5.3f'%float(0.01),
                                  showvalue=0,
                                  command=setAoRadius)
        self.aoRadiusSlider.set(0.01)
        self.aoRadiusSlider.pack(anchor=tk.W, fill=tk.X)

        def updateAmbientOcclusion(e):
            self.shading.ambient_occlusion(float(e))
            self.ambient_occlusion.config(label='Ambient Occlusion: %4.2f'%float(self.shading._ao))
            self.updateImage(only_foc_changed=False, clear_fps=False)
        self.ambient_occlusion = tk.Scale(renderFrame, 
                                  from_=0.0, to=1.0, 
                                  orient=tk.HORIZONTAL, 
                                  resolution=0.01,
                                  showvalue=0,
                                  label='Ambient Occlusion: %4.2f'%float(self.shading._ao),
                                  command=updateAmbientOcclusion)
        self.ambient_occlusion.set(self.shading._ao)
        self.ambient_occlusion.pack(anchor=tk.W, fill=tk.X)

        renderFrame.pack(fill=tk.X)

        # Create direct renderer
        if RENDERER_DIRECT_WRITE:
            self.renderer_ = inference.DirectRenderer(opt.rendererDirectGPU)

    def initRenderer(self, file, update=False):
        if file is None:
            file = filedialog.askopenfilename(
                initialdir=DEFUALT_DATA_DIR if self.last_renderer_dir is None else self.last_renderer_dir,
                title="Select input file",
                filetypes = INPUT_FILETYPES)
            if file is None or not os.path.exists(file):
                return
            self.last_renderer_dir = os.path.dirname(file)

        if RENDERER_DIRECT_WRITE:
            self.renderer_.load(file)
        else:
            if self.renderer_ is not None:
                self.renderer_.close()
            renderer = opt.rendererCPU if file.endswith('vdb') else opt.rendererGPU
            self.renderer_ = inference.Renderer(renderer, file, self.material, self.camera)
        self.input_name = os.path.splitext(os.path.basename(file))[0]
        if update:
            self.updateImage(only_foc_changed=False, clear_fps=True)

    def initChannels(self):
        self.channel_mode = tk.IntVar()
        self.channel_mode.set(GUI.CHANNEL_COLOR)
        self.channel_mode.trace_add('write', lambda a,b,c : self.updateImage(only_foc_changed=False, clear_fps=True))
        tk.Radiobutton(self.channelsFrame, text="Mask", variable=self.channel_mode, value=GUI.CHANNEL_MASK).pack(anchor=tk.W)
        tk.Radiobutton(self.channelsFrame, text="Normal", variable=self.channel_mode, value=GUI.CHANNEL_NORMAL).pack(anchor=tk.W)
        tk.Radiobutton(self.channelsFrame, text="Depth (normalized)", variable=self.channel_mode, value=GUI.CHANNEL_DEPTH).pack(anchor=tk.W)
        tk.Radiobutton(self.channelsFrame, text="AO", variable=self.channel_mode, value=GUI.CHANNEL_AO).pack(anchor=tk.W)
        tk.Radiobutton(self.channelsFrame, text="Flow", variable=self.channel_mode, value=GUI.CHANNEL_FLOW).pack(anchor=tk.W)
        tk.Radiobutton(self.channelsFrame, text="Color", variable=self.channel_mode, value=GUI.CHANNEL_COLOR).pack(anchor=tk.W)
        self.channelsFrame.pack(fill=tk.X)

    def initNetwork(self, files, update=False):
        if files is None:
            files = filedialog.askopenfilenames(
                initialdir=DEFAULT_MODEL_DIR if self.last_network_dir is None else self.last_network_dir,
                title="Select superresolution networks",
                filetypes = (("Pytorch checkpoints", "*.pth"),))
            if files is not None and len(files)>0:
                self.last_network_dir = os.path.dirname(files[0])

        print("Load model(s)")
        self.models_ = [inference.LoadedModel(file, device, upscale_factor) for file in files]
        self.previous_raw_frame = None

        for child in self.superresolutionFrame.winfo_children():
            child.destroy()
        self.render_mode = tk.IntVar()
        self.render_mode.set(GUI.MODE_NEAREST)
        self.render_mode.trace_add('write', lambda a,b,c : self.updateImage(only_foc_changed=False, clear_fps=True))
        tk.Radiobutton(self.superresolutionFrame, text="Nearest", variable=self.render_mode, value=GUI.MODE_NEAREST).pack(anchor=tk.W)
        tk.Radiobutton(self.superresolutionFrame, text="Bilinear", variable=self.render_mode, value=GUI.MODE_BILINEAR).pack(anchor=tk.W)
        tk.Radiobutton(self.superresolutionFrame, text="Bicubic", variable=self.render_mode, value=GUI.MODE_BICUBIC).pack(anchor=tk.W)
        #tk.Radiobutton(self.superresolutionFrame, text="Normal", variable=self.render_mode, value=GUI.MODE_NORMAL).pack(anchor=tk.W)
        #tk.Radiobutton(self.superresolutionFrame, text="Flow", variable=self.render_mode, value=GUI.MODE_FLOW).pack(anchor=tk.W)
        #tk.Radiobutton(self.superresolutionFrame, text="Previous Warped", variable=self.render_mode, value=GUI.MODE_PREVIOUS_WARPED).pack(anchor=tk.W)
        tk.Radiobutton(self.superresolutionFrame, text="Ground Truth", variable=self.render_mode, value=GUI.MODE_GROUND_TRUTH).pack(anchor=tk.W)
        for i,model in enumerate(self.models_, 0):
            tk.Radiobutton(self.superresolutionFrame, text=model.name, variable=self.render_mode, value=GUI.MODE_SUPERRESOLUTION+i).pack(anchor=tk.W)

        self.superresolutionFrame.pack(fill=tk.X)

        if update:
            self.updateImage(only_foc_changed=False, clear_fps=False)

    def initSuperresOptions(self):
        self.temporal_coherence = tk.IntVar()
        self.temporal_coherence.set(1)
        self.temporal_coherence.trace_add('write', lambda a,b,c : self.updateImage(only_foc_changed=False, clear_fps=False))
        tk.Checkbutton(self.superresolutionOptions, text="Temporal Coherence", variable=self.temporal_coherence).pack(anchor=tk.W)

        self.masking = tk.IntVar()
        self.masking.set(0)
        self.masking.trace_add('write', lambda a,b,c : self.updateImage(only_foc_changed=False, clear_fps=False))
        tk.Checkbutton(self.superresolutionOptions, text="Masking", variable=self.masking).pack(anchor=tk.W)

        # temporal post-smoothing
        INITIAL_TEMPORAL_POST_SMOOTHING = 0
        def setPostSmoothing(e):
            self.temporalPostSmoothingSlider.config(label='Post-Smooth: %d%%'%int(e))
            self.updateImage(only_foc_changed=False, clear_fps=False)
        self.temporalPostSmoothingSlider = tk.Scale(self.superresolutionOptions, 
                                  from_=0, to=90, 
                                  orient=tk.HORIZONTAL, 
                                  resolution=1,
                                  label='Post-Smooth: %d%%'%INITIAL_TEMPORAL_POST_SMOOTHING,
                                  showvalue=0,
                                  command=setPostSmoothing)
        self.temporalPostSmoothingSlider.set(INITIAL_TEMPORAL_POST_SMOOTHING)
        self.temporalPostSmoothingSlider.pack(anchor=tk.W, fill=tk.X)

        self.superresolutionOptions.pack(fill=tk.X)

    #@profile
    def setImage(self, img):
        self.pilImage = pl.Image.fromarray(np.clip((img*255).transpose((1, 0, 2)), 0, 255).astype(np.uint8))
        self.tikImage = ImageTk.PhotoImage(self.pilImage)
        self.panel.configure(image=self.tikImage)

    def StartMove(self, event):
        self.x = event.x
        self.y = event.y
        self.camera.startMove()

    def StopMove(self, event):
        self.x = None
        self.y = None
        self.camera.stopMove()

    def OnMouseMotion(self, event):
        self.foc_center = (event.x, event.y)
        self.updateImage(only_foc_changed=True, clear_fps=False)

    def OnMouseDrag(self, event):
        self.foc_center = (event.x, event.y)
        deltax = event.x - self.x
        deltay = event.y - self.y
        self.camera.move(deltax, deltay)
        self.updateImage(only_foc_changed=False, clear_fps=False)

    def OnMouseWheel(self, event):
        self.camera.zoom(event.delta / 120)
        #if event.num == 5 or event.delta == -120:
        #    self.camera.zoom(-1)
        #elif event.num == 4 or event.delta == 120:
        #    self.camera.zoom(+1)
        self.updateImage(only_foc_changed=False, clear_fps=False)

    #@profile
    def focGetBoundsAndMask(self):
        """
        Returns two variables for Focus-of-context:
        the viewport boundary as a tuple (minX, minY, maxX, maxY)
        the blending mask as a np array of shape (1, resY, resX)
          with entry=1 for high-detail-rendering and =0 for upsampling
        All indices are in high-res
        """
        assert self.enable_foc.get()==1

        window_half_width = self.foc_window_size
        window_half_height = window_half_width
        foc_x, foc_y = self.foc_center

        viewport = (max(0, foc_x-window_half_width),
                    max(0, foc_y-window_half_height),
                    min(opt.resX*upscale_factor, foc_x+window_half_width),
                    min(opt.resY*upscale_factor, foc_y+window_half_height))

        outer_radius = self.foc_window_size
        inner_radius = max(0, self.foc_window_size - self.foc_blur_radius)
        def mask_fun(x, y):
            r = np.sqrt(np.square(x-foc_y) + np.square(y-foc_x))
            return np.clip((r-outer_radius)/(inner_radius-outer_radius), 0, 1)
        xaxis = np.linspace(0, opt.resY*upscale_factor-1, opt.resY*upscale_factor, dtype=np.float32)
        yaxis = np.linspace(0, opt.resX*upscale_factor-1, opt.resX*upscale_factor, dtype=np.float32)
        mask = mask_fun(xaxis[:,None], yaxis[None,:])
        mask = mask[np.newaxis,:,:]

        return viewport, torch.from_numpy(mask).to(device)

    def performSuperresolution(self, image, background):
        """
        Performs superresolution
         image: current low-res image on the device
         background: background color (array)
         Reads self.previous_raw_frame (numpy)
        """

        model_idx = self.render_mode.get() - GUI.MODE_SUPERRESOLUTION
        if self.temporal_coherence.get() == 0:
            self.previous_raw_frame = None
        self.shading.inverse_ao = self.models_[model_idx].inverse_ao

        if self.models_[model_idx].unshaded:
            # unshaded input

            torch.cuda.synchronize()
            t0 = time.time()
            imageRaw = self.models_[model_idx].inference(image, self.previous_raw_frame)
            torch.cuda.synchronize()
            elapsed_model_time = time.time()-t0

            imageRaw = torch.cat([
                torch.clamp(imageRaw[:,0:1,:,:], -1, +1),
                ScreenSpaceShading.normalize(imageRaw[:,1:4,:,:], dim=1),
                torch.clamp(imageRaw[:,4:,:,:], 0, 1)
                ], dim=1)
            self.previous_raw_frame = imageRaw

            #image = image.transpose((2, 1, 0))
            #cv.resize(image, dsize=None, fx=upscale_factor, fy=upscale_factor, interpolation=cv.INTER_LINEAR)
            image = F.interpolate(image, scale_factor=upscale_factor, mode='bilinear')
            #image = image.transpose((2, 1, 0))
            base_mask = image[:,3:4,:,:].clone()
            image[:,0:3,:,:] = self.shading(imageRaw)
            image[:,3:8,:,:] = imageRaw[:,0:-1,:,:]
            image[:,10,:,:] = imageRaw[:,-1,:,:]
            #image[[3,4,5,6,7,10],:,:] = imageRawCpu

        else:
            #TODO: All ops on the GPU

            # shaded input
            image_with_color = torch.cat([
                image[:,0:3,:,:],
                image[:,3:4,:,:]*0.5+0.5, #transform mask back to [0,1]
                image[:,4:,:,:]], dim=1) 

            torch.cuda.synchronize()
            t0 = time.time()
            imageRaw = self.models_[model_idx].inference(image_with_color, self.previous_raw_frame)
            torch.cuda.synchronize()
            elapsed_model_time = time.time()-t0
            
            image = F.interpolate(image, scale_factor=upscale_factor, mode='bilinear')
            base_mask = image[:,3:4,:,:].clone()
            image[:,0:3,:,:] = torch.clamp(imageRaw, 0, 1)

        if self.masking.get() == 1:
            DILATION = 1
            KERNEL = np.ones((3,3), np.float32)
            mask = (base_mask*0.5+0.5)
            #mask = cv.dilate((image[3,:,:]*0.5+0.5).transpose((1, 0)), KERNEL, iterations = DILATION)
            #mask = cv.resize(mask, dsize=None, fx=upscale_factor, fy=upscale_factor, interpolation=cv.INTER_LINEAR)
            image = background[0] + mask * (image - background[0])
        self.upres_time = elapsed_model_time

        return image

    #@profile
    def updateImage(self, only_foc_changed, clear_fps):
        if self.input_name is None:
            return # no dataset loaded
        if clear_fps:
            self.fps.clear()
            self.previous_raw_frame = None
            self.previous_rgb_images = None
        if self.renderer_ is None:
            return

        t1 = time.time()

        # prepare shading
        self.shading.inverse_ao = False
        background = None
        if self.background_mode.get() == GUI.BACKGROUND_BLACK:
            background = np.array([0,0,0])
        else:
            background = np.array([1,1,1])
        self.shading.background(background)

        # general renderer settings
        currentOrigin = self.camera.getOrigin()
        currentLookAt = self.camera.getLookAt()
        currentUp = self.camera.getUp()
        self.renderer_.send_command("cameraOrigin", "%5.3f,%5.3f,%5.3f"%(currentOrigin[0], currentOrigin[1], currentOrigin[2]))
        self.renderer_.send_command("cameraLookAt", "%5.3f,%5.3f,%5.3f"%(currentLookAt[0], currentLookAt[1], currentLookAt[2]))
        self.renderer_.send_command("cameraUp", "%5.3f,%5.3f,%5.3f"%(currentUp[0], currentUp[1], currentUp[2]))
        self.renderer_.send_command("cameraFoV", "%.3f"%self.camera_fov)
        self.renderer_.send_command("isovalue", "%5.3f"%float(self.isoslider.get()))

        ####################################################
        # render fullscreen version
        ####################################################
        if not only_foc_changed:
            ####################################################
            # CALL RENDERER
            ####################################################
            # send remaining settings
            if self.render_mode.get() == GUI.MODE_GROUND_TRUTH:
                resX = opt.resX * upscale_factor
                resY = opt.resY * upscale_factor
            else:
                resX = opt.resX
                resY = opt.resY
            self.renderer_.send_command("resolution", "%d,%d"%(resX, resY))
            self.renderer_.send_command("viewport", "%d,%d,%d,%d"%(0,0,resX, resY))
            self.renderer_.send_command("aoradius", "%5.3f"%float(self.aoRadiusSlider.get()))
            if self.render_mode.get() >= GUI.MODE_SUPERRESOLUTION:
                self.renderer_.send_command("aosamples", "0")
            else:
                self.renderer_.send_command("aosamples", "%d"%int(self.aoSamplesSlider.get()))
            # call and fetch result
            if RENDERER_DIRECT_WRITE:
                # for testing, create new tensors and copy directly to Numpy
                shared_tensor = torch.empty((resY, resX, 12), dtype=torch.float32, device=device)
                shared_tensor.share_memory_()
                assert shared_tensor.is_cuda
                #print(shared_tensor.data_ptr())
                self.render_time = self.renderer_.render_direct(shared_tensor)
                image = shared_tensor.permute(2,0,1)
            else:
                self.renderer_.send_command("render\n")
                image = self.renderer_.read_image(resX, resY)
                self.render_time = self.renderer_.get_time()
                image = torch.from_numpy(image).to(device)
            new_title = self.input_name + "  -  render time: %5.3fs" % self.render_time
            self.upres_time = 0

            # preprocessing
            image = torch.unsqueeze(image, 0)
            self.original_image = image.clone()
            image = torch.cat((
                image[:,0:3,:,:],
                image[:,3:4,:,:]*2-1, #transform mask into -1,+1
                image[:,4: ,:,:]), dim=1)
            image_shaded_input = torch.cat((image[:,3:4,:,:], image[:,4:8,:,:], image[:,10:11,:,:]), dim=1)
            image_shaded = torch.clamp(self.shading(image_shaded_input), 0, 1)
            image[:,0:3,:,:] = image_shaded
            # image now contains all channels:
            # 0:3 - color (shaded)
            # 3:$ - mask in -1,+1
            # 4:7 - normal
            # 7:8 - depth
            # 8:10 - flow
            # 10:11 - AO

            ####################################################
            # Perform superresolution
            ####################################################
            if self.render_mode.get() == GUI.MODE_NEAREST:
                #image = image.transpose((2, 1, 0))
                #image = cv.resize(image, dsize=None, fx=upscale_factor, fy=upscale_factor, interpolation=cv.INTER_NEAREST)
                #image = image.transpose((2, 1, 0))
                image = F.interpolate(image, scale_factor=upscale_factor, mode='nearest')
                self.previous_raw_frame = None
            elif self.render_mode.get() == GUI.MODE_BILINEAR:
                #image = image.transpose((2, 1, 0))
                #image = cv.resize(image, dsize=None, fx=upscale_factor, fy=upscale_factor, interpolation=cv.INTER_LINEAR)
                #image = image.transpose((2, 1, 0))
                image = F.interpolate(image, scale_factor=upscale_factor, mode='bilinear')
                self.previous_raw_frame = None
            elif self.render_mode.get() == GUI.MODE_BICUBIC:
                #image = image.transpose((2, 1, 0))
                #image = cv.resize(image, dsize=None, fx=upscale_factor, fy=upscale_factor, interpolation=cv.INTER_CUBIC)
                #image = image.transpose((2, 1, 0))
                image = F.interpolate(image, scale_factor=upscale_factor, mode='bicubic')
                self.previous_raw_frame = None
            elif self.render_mode.get() == GUI.MODE_GROUND_TRUTH:
                # do nothing, image is already in high-res
                self.previous_raw_frame = None
            else:
                image = self.performSuperresolution(image, background)

            #store that image for use if only focus-of-context changes
            self.upres_frame = image
        else:
            # only focus-of-context changed, restore stored image
            image = self.upres_frame

        ####################################################
        # Perform focus-of-context
        ####################################################
        if self.enable_foc.get()==1:
            viewport, mask = self.focGetBoundsAndMask()
            # render ground truth only in the viewport
            resX = opt.resX * upscale_factor
            resY = opt.resY * upscale_factor
            self.renderer_.send_command("resolution", "%d,%d"%(resX, resY))
            self.renderer_.send_command("viewport", "%d,%d,%d,%d"%viewport)
            self.renderer_.send_command("aosamples", "%d\n"%int(self.aoSamplesSlider.get()))
             # call and fetch result
            if RENDERER_DIRECT_WRITE:
                # for testing, create new tensors and copy directly to Numpy
                shared_tensor = torch.empty((resY, resX, 12), dtype=torch.float32, device=device)
                shared_tensor.share_memory_()
                assert shared_tensor.is_cuda
                print(shared_tensor.data_ptr())
                self.render_time + self.renderer_.render_direct(shared_tensor)
                foc_image = shared_tensor.permute(2,0,1)
            else:
                self.renderer_.send_command("render\n")
                foc_image = self.renderer_.read_image(resX, resY)
                self.render_time += self.renderer_.get_time()

            # preprocessing
            foc_image = torch.unsqueeze(foc_image, 0)
            foc_image = torch.cat((
                foc_image[:,0:3,:,:],
                foc_image[:,3:4,:,:]*2-1, #transform mask into -1,+1
                foc_image[:,4: ,:,:]), dim=1)
            foc_image_shaded_input = torch.cat((foc_image[:,3:4,:,:], foc_image[:,4:8,:,:], foc_image[:,10:11,:,:]), dim=1)
            foc_image_shaded = torch.clamp(self.shading(foc_image_shaded_input), 0, 1)
            foc_image[:,0:3,:,:] = foc_image_shaded

            # blend it
            image = mask * foc_image + (1-mask) * image

        ####################################################
        # CHANNEL SELECTION
        ####################################################
        if self.channel_mode.get() == GUI.CHANNEL_MASK:
            imageRGB = torch.cat((image[:,3:4,:,:], image[:,3:4,:,:], image[:,3:4,:,:]), dim=1) #np.tile((image[3,:,:].transpose((1,0)))[:,:,np.newaxis], (1,1,3))
        elif self.channel_mode.get() == GUI.CHANNEL_NORMAL:
            imageRGB = image[:,4:7,:,:] * 0.5 + 0.5
        elif self.channel_mode.get() == GUI.CHANNEL_DEPTH:
            depthVal = image[:,7:8,:,:]
            depthForBounds = self.original_image[:,7:8,:,:]
            maxDepth = torch.max(depthForBounds)
            minDepth = torch.min(depthForBounds + torch.le(depthForBounds, 1e-5).type_as(depthForBounds))
            #print("minDepth:",minDepth,"maxDepth:",maxDepth)
            depthVal = (depthVal - minDepth) / (maxDepth - minDepth)
            imageRGB = torch.cat((depthVal, depthVal, depthVal), dim=1) #np.tile((image[7,:,:].transpose((1,0)))[:,:,np.newaxis], (1,1,3))
        elif self.channel_mode.get() == GUI.CHANNEL_AO:
            imageRGB = torch.cat((image[:,10:11,:,:], image[:,10:11,:,:], image[:,10:11,:,:]), dim=1) #np.tile((image[10,:,:].transpose((1,0)))[:,:,np.newaxis], (1,1,3))
        elif self.channel_mode.get() == GUI.CHANNEL_FLOW:
            original_image_cpu = self.original_image.cpu().numpy()[0]
            flow_inpaint = np.stack((
                cv.inpaint(original_image_cpu[8,:,:], np.uint8(original_image_cpu[3,:,:]==0), 3, cv.INPAINT_NS),
                cv.inpaint(original_image_cpu[9,:,:], np.uint8(original_image_cpu[3,:,:]==0), 3, cv.INPAINT_NS),
                np.zeros((original_image_cpu.shape[1], original_image_cpu.shape[2]))), axis=0).astype(np.float32)
            flow_inpaint = torch.unsqueeze(torch.from_numpy(flow_inpaint).to(device), 0)
            imageRGB = (flow_inpaint * 10 + 0.5)
            imageRGB = F.interpolate(imageRGB, scale_factor=upscale_factor, mode='bilinear')
            #imageRGB = cv.resize(imageRGB, dsize=None, fx=upscale_factor, fy=upscale_factor, interpolation=cv.INTER_LINEAR)
        elif self.channel_mode.get() == GUI.CHANNEL_COLOR:
            imageRGB = image[:,0:3,:,:]#.transpose((2,1,0))

        ####################################################
        # FINAL IMAGE PROCESSING
        ####################################################

        # exponential blending
        if self.previous_rgb_images is None \
                or self.temporalPostSmoothingSlider.get()==0 \
                or self.render_mode.get() == GUI.MODE_GROUND_TRUTH:
            pass # do nothing
        else:
            original_image_cpu = self.original_image.cpu().numpy()[0]
            flow_inpaint = np.stack((
                cv.inpaint(original_image_cpu[8,:,:], np.uint8(original_image_cpu[3,:,:]==0), 3, cv.INPAINT_NS),
                cv.inpaint(original_image_cpu[9,:,:], np.uint8(original_image_cpu[3,:,:]==0), 3, cv.INPAINT_NS)), axis=0).astype(np.float32)
            flow = torch.unsqueeze(torch.from_numpy(flow_inpaint), dim=0).to(device)
            previous_warped = models.VideoTools.warp_upscale(
                torch.unsqueeze(torch.from_numpy(self.previous_rgb_images.transpose((2, 1, 0))), dim=0).to(device), flow, 4)

            factor = float(self.temporalPostSmoothingSlider.get()) / 100.0
            imageRGB = factor * previous_warped + (1 - factor) * imageRGB

        # finally, copy to cpu
        imageRGB = imageRGB.cpu().numpy()[0].transpose((2,1,0))
        self.previous_rgb_images = imageRGB

        ####################################################
        # OUTPUT
        ####################################################

        # update title
        t2 = time.time()
        if not only_foc_changed or self.enable_foc.get()==1:
            if len(self.fps) > 10:
                self.fps.pop(0)
            self.fps.append(t2-t1)
        if len(self.fps)==0:
            fps = 0.0
        else:
            fps = 1.0 / ((sum(self.fps) + 1e-7) / len(self.fps))
        new_title = self.input_name + \
            "  -  render time: %5.3fs" % self.render_time + \
            "  -  superresolution time: %5.3fs" % self.upres_time + \
             "  -> total %.3f FPS" % fps
        self.title(new_title)

        # send image and store it
        self.setImage(imageRGB)
        self.previous_frame = np.transpose(imageRGB, (2, 1, 0))

    def saveScreenshot(self):
        import json, time

        SCREENSHOT_DIR = "screenshots"
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)

        # collect information
        info = dict({})

        model_idx = self.render_mode.get() - GUI.MODE_SUPERRESOLUTION
        if self.render_mode.get() == GUI.MODE_NEAREST: info['model'] = 'nearest'
        elif self.render_mode.get() == GUI.MODE_BILINEAR: info['model'] = 'bilinear'
        elif self.render_mode.get() == GUI.MODE_BICUBIC: info['model'] = 'bicubic'
        #elif self.render_mode.get() == GUI.MODE_NORMAL: info['model'] = 'normal'
        #elif self.render_mode.get() == GUI.MODE_FLOW: info['model'] = 'flow'
        #elif self.render_mode.get() == GUI.MODE_PREVIOUS_WARPED: info['model'] = 'previous'
        elif self.render_mode.get() == GUI.MODE_GROUND_TRUTH: info['model'] = 'gt'
        else: info['model'] = self.models_[model_idx].name

        if self.channel_mode.get() == GUI.CHANNEL_MASK: info['channel'] = 'mask'
        elif self.channel_mode.get() == GUI.CHANNEL_NORMAL: info['channel'] = 'normal'
        elif self.channel_mode.get() == GUI.CHANNEL_DEPTH: info['channel'] = 'depth'
        elif self.channel_mode.get() == GUI.CHANNEL_AO: info['channel'] = 'ao'
        elif self.channel_mode.get() == GUI.CHANNEL_FLOW: info['channel'] = 'flow'
        else: info['channel'] = 'color'

        info['data'] = self.input_name
        info['timestamp'] = time.strftime('%mm%dd-%Hh%Mm%Ss')
        info['iso'] = self.isoslider.get()
        info['shading'] = {
            'ambient_light' : list(self.ambient_light_color),
            'diffuse_light' : list(self.diffuse_light_color),
            'specular_light' : list(self.specular_light_color),
            'specular_exponent' : self.shading._specular_exponent,
            'material_color' : list(self.material_color),
        }
        info['ao'] = {
            'samples' : self.aoSamplesSlider.get(),
            'radius' : self.aoRadiusSlider.get(),
            'strength' : self.shading._ao
        }

        # generate name
        timestamp = ""
        name = info['data']+"."+info['model']+"."+info['channel']+"."+info['timestamp']+".png"
        name = os.path.join(SCREENSHOT_DIR, name)

        # save
        self.pilImage.save(name)
        with open(name+".json", 'w') as outfile:  
            json.dump(info, outfile, indent=4, sort_keys=True)
        print('Screenshot saved to', name)

    def close(self):
        if self.renderer_ is not None:
            self.renderer_.close()
            self.renderer_ = None

gui = GUI()

gui.initRenderer(opt.input)
modelNames = opt.model.split(':') if opt.model is not None else None
gui.initNetwork(modelNames)
gui.updateImage(only_foc_changed=False, clear_fps=True)

gui.mainloop()

gui.close()