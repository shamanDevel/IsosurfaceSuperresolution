import numpy as np
import cv2 as cv
import torch
from models import VideoTools
import os.path
from utils import initialImage

class LoadedModel:
    def __init__(self, name, device, upscale_factor):
        super().__init__()
        self.name = os.path.splitext(os.path.basename(name))[0]
        self.device = device
        self.upscale_factor = upscale_factor
        print(self.name+':')

        checkpoint = torch.load(name)
        self.parameters = checkpoint.get('parameters', dict())
        if not isinstance(self.parameters, dict):
            self.parameters = vars(self.parameters) # namespace object
        self.model = checkpoint['model']
        self.model.to(device)
        self.model.train(False)
        print(self.model)

        #find first module
        first_module = self.model
        while True:
            it = first_module.children()
            try:
                o = next(it)
            except StopIteration:
                break
            first_module = o
        print('The first module in the network is:', first_module)
        self.input_channels = first_module.in_channels
        if self.input_channels == 5 + 6 * (self.upscale_factor**2) or \
                self.parameters.get('unshaded', False):
            self.unshaded = True
            print("Network runs in unshaded mode")
        else:
            self.unshaded = False
            self.input_single_channels = self.input_channels - 3 * (self.upscale_factor**2)
            if self.input_single_channels >= 7:
                self.has_normal = True
                self.input_single_channels -= 3
            else:
                self.has_normal = False
            if self.input_single_channels >= 5:
                self.has_depth = True
                self.input_single_channels -= 1
            else:
                self.has_depth = False
            print('Number of input channels:', self.input_channels,
                 ', has_normal:', self.has_normal, 
                 ', has_depth:', self.has_depth)

        # Read mode for initial image
        if not "initialImage" in self.parameters:
            if self.unshaded:
                self.initial_image_mode = "input"
            else:
                self.initial_image_mode = "zero"
        else:
            self.initial_image_mode = self.parameters['initialImage']
        print('initial image mode:', self.initial_image_mode)

        # Read other settings
        self.inverse_ao = self.parameters.get('aoInverted', False)

    def inference(self, current_low, prev_high):
        """
        Performs the superresolution.
        current_low: low-resolution input from the renderer, 10 channels (RGB, mask, normal, depth, flow), GPU. Format: (B,C,H,W)
        prev_high: RGB-image of the previous inference result
        """
        with torch.no_grad():
            current_low_cpu = current_low.cpu().numpy()[0]
            # compute flow
            flow_inpaint = np.stack((
                cv.inpaint(current_low_cpu[8,:,:], np.uint8(current_low_cpu[3,:,:]==0), 3, cv.INPAINT_NS),
                cv.inpaint(current_low_cpu[9,:,:], np.uint8(current_low_cpu[3,:,:]==0), 3, cv.INPAINT_NS)), axis=0).astype(np.float32)
            flow = torch.unsqueeze(torch.from_numpy(flow_inpaint), dim=0).to(self.device)
            #input
            if self.unshaded:
                input = torch.cat((current_low[:,3:4,:,:]*2-1, current_low[:,4:8,:,:]), dim=1)
                if prev_high is None:
                    previous_warped = initialImage(input, 6, 
                                             self.initial_image_mode, 
                                             self.inverse_ao,
                                             self.upscale_factor).to(self.device)
                else:
                    previous_warped = VideoTools.warp_upscale(
                        prev_high.to(self.device), 
                        flow, 
                        self.upscale_factor,
                        special_mask = True)
            else:
                if self.has_normal and self.has_depth:
                    input = torch.clamp(current_low[:,0:8,:,:], 0, 1)
                elif self.has_normal: #no depth
                    input = current_low[:,0:7,:,:]
                elif self.has_depth: #no normal
                    input = torch.cat((current_low[:,0:4,:,:], current_low[:,7:8,:,:]), dim=1)
                else: #only color+mask
                    input = current_low[:,0:4,:,:]
                if prev_high is None:
                    #prev_high = np.zeros(
                    #    (3, input.shape[2]*self.upscale_factor, input.shape[3]*self.upscale_factor),
                    #    dtype=current_low.dtype)
                    prev_high = initialImage(input, 3, self.initial_image_mode, self.upscale_factor)
                previous_warped = VideoTools.warp_upscale(
                    prev_high.to(self.device), 
                    flow,
                    self.upscale_factor,
                    special_mask = False)
            previous_warped_flattened = VideoTools.flatten_high(previous_warped, self.upscale_factor)
            # run the network
            single_input = torch.cat((input, previous_warped_flattened), dim=1)
            prediction, _ = self.model(single_input)
            
        return prediction