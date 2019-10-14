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

INITIAL_DIR = "..\\..\\data\\volumes\\rendering"
UPSCALE = 4

class GUI(tk.Tk):

    MODE_LOW = 1
    MODE_HIGH = 2
    MODE_FLOW = 3

    CHANNEL_MASK = 1
    CHANNEL_NORMAL = 2
    CHANNEL_DEPTH = 3
    CHANNEL_AO = 4
    CHANNEL_COLOR = 5

    def __init__(self):
        tk.Tk.__init__(self)
        self.title('Dataset Viewer')
        self.input_name = None

        # members to be filled later
        self.pilImage = None
        self.tikImage = None
        self.current_high = None
        self.current_low = None
        self.current_flow = None
        self.dataset_folder = None
        self.last_folder = INITIAL_DIR
        self.entries = []
        self.num_frames = 0
        self.selected_time = 0

        # root
        self.root_panel = tk.Frame(self)
        self.root_panel.pack(side="bottom", fill="both", expand="yes")
        self.panel = tk.Label(self.root_panel)
        self.black = np.zeros((512, 512, 3))
        self.setImage(self.black)
        self.panel.pack(side="left", fill="both", expand="yes")

        options1 = tk.Label(self.root_panel)

        # Input
        inputFrame = ttk.LabelFrame(options1, text="Input", relief=tk.RIDGE)
        tk.Button(inputFrame, text="Open Folder", command=lambda : self.openFolder()).pack()
        self.dataset_entry = 0
        self.dataset_entry_slider = tk.Scale(
            inputFrame,
            from_=0, to=0,
            orient=tk.HORIZONTAL,
            resolution=1,
            label="Entry",
            showvalue=0,
            command=lambda e: self.setEntry(int(e)))
        self.dataset_entry_slider.pack(anchor=tk.W, fill=tk.X)
        self.dataset_time = 0
        self.dataset_time_slider = tk.Scale(
            inputFrame,
            from_=0, to=0,
            orient=tk.HORIZONTAL,
            resolution=1,
            label="Time",
            showvalue=0,
            command=lambda e: self.setTime(int(e)))
        self.dataset_time_slider.pack(anchor=tk.W, fill=tk.X)
        inputFrame.pack(fill=tk.X)

        # Mode
        modeFrame = ttk.LabelFrame(options1, text="Mode", relief=tk.RIDGE)
        self.mode = tk.IntVar()
        self.mode.set(GUI.MODE_LOW)
        self.mode.trace_add('write', lambda a,b,c : self.updateImage())
        tk.Radiobutton(modeFrame, text="Low", variable=self.mode, value=GUI.MODE_LOW).pack(anchor=tk.W)
        tk.Radiobutton(modeFrame, text="High", variable=self.mode, value=GUI.MODE_HIGH).pack(anchor=tk.W)
        tk.Radiobutton(modeFrame, text="Flow", variable=self.mode, value=GUI.MODE_FLOW).pack(anchor=tk.W)
        modeFrame.pack(fill=tk.X)

        # Channel
        channelsFrame = ttk.LabelFrame(options1, text="Channel", relief=tk.RIDGE)
        self.channel_mode = tk.IntVar()
        self.channel_mode.set(GUI.CHANNEL_COLOR)
        self.channel_mode.trace_add('write', lambda a,b,c : self.updateImage())
        tk.Radiobutton(channelsFrame, text="Mask", variable=self.channel_mode, value=GUI.CHANNEL_MASK).pack(anchor=tk.W)
        tk.Radiobutton(channelsFrame, text="Normal", variable=self.channel_mode, value=GUI.CHANNEL_NORMAL).pack(anchor=tk.W)
        tk.Radiobutton(channelsFrame, text="Depth", variable=self.channel_mode, value=GUI.CHANNEL_DEPTH).pack(anchor=tk.W)
        tk.Radiobutton(channelsFrame, text="AO", variable=self.channel_mode, value=GUI.CHANNEL_AO).pack(anchor=tk.W)
        tk.Radiobutton(channelsFrame, text="Color", variable=self.channel_mode, value=GUI.CHANNEL_COLOR).pack(anchor=tk.W)
        channelsFrame.pack(fill=tk.X)

        # Shading
        self.shading = ScreenSpaceShading('cpu')
        self.shading.fov(30)
        self.ambient_light_color = np.array([0.1,0.1,0.1])
        self.shading.ambient_light_color(self.ambient_light_color)
        self.diffuse_light_color = np.array([0.8, 0.8, 0.8])
        self.shading.diffuse_light_color(self.diffuse_light_color)
        self.specular_light_color = np.array([0.02, 0.02, 0.02])
        self.shading.specular_light_color(self.specular_light_color)
        self.shading.specular_exponent(16)
        self.shading.light_direction(np.array([0.1,0.1,1.0]))
        self.material_color = np.array([1.0,1.0,1.0])#[1.0, 0.3, 0.0])
        self.shading.material_color(self.material_color)
        self.shading.ambient_occlusion(0.5)
        self.shading.background(np.array([1.0, 1.0, 1.0]))

        # Save
        tk.Button(options1, text="Save Image", command=lambda : self.saveImage()).pack()
        self.saveFolder = "/"

        options1.pack(side="left")

    def openFolder(self):
        dataset_folder = filedialog.askdirectory(initialdir=self.last_folder)
        print(dataset_folder)
        if dataset_folder is not None:
            self.dataset_folder = str(dataset_folder)
            print("New folder selected:", self.dataset_folder)
            self.last_folder = self.dataset_folder
            # find number of entries
            self.current_low = None
            self.current_high = None
            self.current_flow = None
            entries = []
            for i in range(0, 10000):
                file_low = os.path.join(self.dataset_folder, "low_%05d.npy"%i)
                if os.path.isfile(file_low):
                    entries.append((
                        file_low,
                        os.path.join(self.dataset_folder, "high_%05d.npy"%i),
                        os.path.join(self.dataset_folder, "flow_%05d.npy"%i)
                        ))
                else:
                    break
            print("Number of entries found:", len(entries))
            self.entries = entries
            self.dataset_entry = 0
            self.dataset_entry_slider.config(to=len(entries))
            self.setEntry(0)

        else:
            print("No folder selected")


    def setEntry(self, entry):
        entry = min(entry, len(self.entries)-1)
        self.dataset_entry_slider.config(label='Entry: %d'%int(entry))
        if len(self.entries)==0:
            self.current_low = None
            self.current_high = None
            self.current_flow = None
        else:
            self.current_low = np.load(self.entries[entry][0])
            self.current_high = np.load(self.entries[entry][1])
            self.current_flow = np.load(self.entries[entry][2])
            self.num_frames = self.current_low.shape[0]
            self.dataset_time_slider.config(to=self.num_frames)
            print("Entry loaded")
            self.setTime(0)

    def setTime(self, entry):
        entry = min(entry, self.num_frames-1)
        self.dataset_time_slider.config(label='Time: %d'%int(entry))
        self.selected_time = entry
        self.updateImage()

    def setImage(self, img):
        self.pilImage = pl.Image.fromarray(np.clip((img*255).transpose((1, 0, 2)), 0, 255).astype(np.uint8))
        self.tikImage = ImageTk.PhotoImage(self.pilImage)
        self.panel.configure(image=self.tikImage)

    def saveImage(self):
        filename =  filedialog.asksaveasfilename(initialdir = self.saveFolder,title = "Save as",filetypes = (("jpeg files","*.jpg"),("png files", "*.png"),("all files","*.*")))
        if len(filename)==0:
            return;
        if len(os.path.splitext(filename)[1])==0:
            filename = filename + ".jpg"
        self.pilImage.save(filename)
        self.saveFolder = os.path.dirname(filename)

    def updateImage(self):
        if self.current_low is None:
            # no image loaded
            self.setImage(self.black)
            return

        def selectChannel(img):
            if self.channel_mode.get() == GUI.CHANNEL_MASK:
                mask = img[0:1,:,:] * 0.5 + 0.5
                return np.concatenate((mask, mask, mask))
            elif self.channel_mode.get() == GUI.CHANNEL_NORMAL:
                return img[1:4,:,:] * 0.5 + 0.5
            elif self.channel_mode.get() == GUI.CHANNEL_DEPTH:
                return np.concatenate((img[4:5,:,:], img[4:5,:,:], img[4:5,:,:]))
            elif self.channel_mode.get() == GUI.CHANNEL_AO:
                if img.shape[0]==6:
                    return np.concatenate((img[5:6,:,:], img[5:6,:,:], img[5:6,:,:]))
                else:
                    return np.zeros((3, img.shape[1], img.shape[2]), dtype=np.float32)
            elif self.channel_mode.get() == GUI.CHANNEL_COLOR:
                shading_input = torch.unsqueeze(torch.from_numpy(img), 0)
                shading_output = self.shading(shading_input)[0]
                return torch.clamp(shading_output, 0, 1).cpu().numpy()

        if self.mode.get() == GUI.MODE_LOW:
            img = self.current_low[self.selected_time,:,:,:]
            img = selectChannel(img)
            img = img.transpose((2, 1, 0))
            img = cv.resize(img, dsize=None, fx=UPSCALE, fy=UPSCALE, interpolation=cv.INTER_NEAREST)
            self.setImage(img)
        elif self.mode.get() == GUI.MODE_HIGH:
            img = self.current_high[self.selected_time,:,:,:]
            img = selectChannel(img)
            img = img.transpose((2, 1, 0))
            self.setImage(img)
        elif self.mode.get() == GUI.MODE_FLOW:
            #img = np.stack((
            #    cv.inpaint(self.current_flow[self.selected_time,0,:,:], np.uint8(self.current_low[self.selected_time,0,:,:]==0), 3, cv.INPAINT_NS),
            #    cv.inpaint(self.current_flow[self.selected_time,1,:,:], np.uint8(self.current_low[self.selected_time,0,:,:]==0), 3, cv.INPAINT_NS),
            #    np.zeros((self.current_flow.shape[2], self.current_flow.shape[3]))), axis=0).astype(np.float32)
            img = np.concatenate(
                (self.current_flow[self.selected_time,0:2,:,:], 
                 np.zeros((1, self.current_flow.shape[2], self.current_flow.shape[3]), dtype=np.float32)),
            )
            img = (img * 10 + 0.5)
            img = img.transpose((2, 1, 0))
            img = cv.resize(img, dsize=None, fx=UPSCALE, fy=UPSCALE, interpolation=cv.INTER_NEAREST)
            self.setImage(img)

if __name__ == "__main__":
    gui = GUI()
    gui.mainloop()
