import torch.utils.data as data

import os.path
import imageio
import collections
import random
import numpy as np
import torch
import cv2 as cv
import time
from console_progressbar import ProgressBar

video_crop_size = 32 # arbitrary chosen, low-res value
                     # chosen so that the high-res patch is of size 128^2

use_augmentation = False

# In new versions, the data generator already performs the preprocessing.
# Hence, we only load the processed arrays.
load_arrays = True 

# Load and argument samples
Sample = collections.namedtuple('Sample', """
index,crop_high,crop_low,augmentation,
ambient_color,diffuse_color,light_direction
""")
DatasetData = collections.namedtuple('DatasetData', """
samples,images_high,images_low,flow_low,input_channels,output_channels,crop_size,num_frames,
""")

MAX_AUGMENTATION_MODE = 4
def data_augmentation(low, high, flow, mode):
    """
    Performs data augmentation (mirroring and rotation).
    Input:
     - low resolution input (TxCxHxW, C=4 (only rgb+mask), C=5 (with depth) C=7 (with normal), C=8 (all)
     - high resolution input (TxCxHhxWh, C=3)
     - flow input (Tx2xHxW)
     - augmentation mode, between zero and MAX_AUGMENTATION_MODE-1
    """
    if not use_augmentation:
        return low, high, flow

    T, C, H, W = low.shape
    assert C==4 or C==5 or C==7 or C==8
    has_normal = True if (C==7 or C==8) else False
    has_depth = True if (C==5 or C==8) else False
    normal_x_idx = 4
    normal_y_idx = 5

    # decode mode
    flipX = True if mode&1 else False
    flipY = True if mode&2 else False

    # special processing for normals and flow needed
    if flipX and flipY:
        low = np.flip(low, axis=(2, 3))
        high = np.flip(high, axis=(2, 3))
        flow = np.flip(flow, axis=(2, 3))
        if has_normal:
            low[:,normal_x_idx,:,:] = -low[:,normal_x_idx,:,:]
            low[:,normal_y_idx,:,:] = -low[:,normal_y_idx,:,:]
        flow[:,0,:,:] = -flow[:,0,:,:]
        flow[:,1,:,:] = -flow[:,1,:,:]
    elif flipX:
        low = np.flip(low, axis=2)
        high = np.flip(high, axis=2)
        flow = np.flip(flow, axis=2)
        if has_normal:
            low[:,normal_x_idx,:,:] = -low[:,normal_x_idx,:,:]
        flow[:,0,:,:] = -flow[:,0,:,:]
    elif flipY:
        low = np.flip(low, axis=3)
        high = np.flip(high, axis=3)
        flow = np.flip(flow, axis=3)
        if has_normal:
            low[:,normal_y_idx,:,:] = -low[:,normal_y_idx,:,:]
        flow[:,1,:,:] = -flow[:,1,:,:]

    return np.ascontiguousarray(low), \
           np.ascontiguousarray(high), \
           np.ascontiguousarray(flow)

def collect_samples_clouds_video(upsampling, opt, deferred_shading=False):
    """
    Collect samples of cloud videos.
    opt is expected to be a dict.
    Output: DatasetData
     - samples: list of Sample
     - images_high: num_frames x output_channels x H*upsampling x W*upsampling
     - images_low: num_frames x input_channels x H x W
     - flow_low: num_frames x 2 x H x W
    """

    number_of_samples = opt['samples']
    number_of_images = opt['numberOfImages']
    use_input_depth = deferred_shading or opt['useInputDepth']
    use_input_normal = deferred_shading or opt['useInputNormal']
    INPUT_PATH_SHADED = opt['inputPathShaded'] or'../../data/clouds/rendering_video/'
    INPUT_PATH_UNSHADED = opt['inputPathUnshaded'] or '../../data/clouds/rendering_video3/'
    inputPath = INPUT_PATH_UNSHADED if deferred_shading else INPUT_PATH_SHADED
    inputExtension = '.exr'

    if load_arrays and deferred_shading:
        # load directly from numpy arrays
        def get_image_name(i,mode,p):
            if mode=='high':
                return os.path.join(p, "high_%05d.npy" % i)
            if mode=='low':
                return os.path.join(p, "low_%05d.npy" % i)
            elif mode=='flow':
                return os.path.join(p, "flow_%05d.npy" % i)

        # Collect number of images and paths
        image_paths = []
        print("dataset path:", inputPath)
        if os.path.isfile(inputPath):
            # input path points to a file where each line is a subdirectory of sets
            with open(inputPath, 'r') as fp:
                while True:
                    line = fp.readline()
                    if line is None or len(line)==0: break
                    p = os.path.join(os.path.dirname(inputPath), line[:-1])
                    print("Check path '%s'"%p)
                    num_images = 0
                    while True:
                        if not os.path.exists(get_image_name(num_images, 'low', p)):
                            break
                        image_paths.append((
                            get_image_name(num_images, 'high', p),
                            get_image_name(num_images, 'low', p),
                            get_image_name(num_images, 'flow', p)
                            ))
                        num_images += 1
        else:
            # input path is directly a folder
            num_images = 0
            while True:
                if not os.path.exists(get_image_name(num_images, 'low', inputPath)):
                    break
                image_paths.append((
                    get_image_name(num_images, 'high', inputPath),
                    get_image_name(num_images, 'low', inputPath),
                    get_image_name(num_images, 'flow', inputPath)
                    ))
                num_images += 1

        num_images = len(image_paths)
        if num_images==0:
            raise ValueError("No image found")
        num_frames = np.load(image_paths[0][1]).shape[0]
        print('Number of images found: %d, each with %d frames' % (num_images, num_frames))
        if number_of_images is not None and number_of_images>0:
            num_images = min(num_images, number_of_images)
            print('But limited to %d images'%number_of_images)

        # load all images
        pg = ProgressBar(num_images, 'Load all images (npy)', length=50)
        images_high = [None]*num_images
        images_low = [None]*num_images
        flow_low = [None]*num_images
        for i in range(num_images):
            pg.print_progress_bar(i)
            images_high[i] = np.load(image_paths[i][0])
            images_low[i] = np.load(image_paths[i][1])
            flow_low[i] = np.load(image_paths[i][2])
        pg.print_progress_bar(num_images)

        input_channels = 5
        output_channels = images_high[0].shape[1]

    else:
        #old version, load images seperately

        def get_image_name(i,j,mode):
            if mode=='high':
                return os.path.join(inputPath, "high_%05d_%05d%s" % (i, j, inputExtension))
            if mode=='highdn':
                return os.path.join(inputPath, "high_%05d_%05d_depth%s" % (i, j, inputExtension))
            elif mode=='low':
                return os.path.join(inputPath, "low_%05d_%05d%s" % (i, j, inputExtension))
            elif mode=='dn':
                return os.path.join(inputPath, "low_%05d_%05d_depth%s" % (i, j, inputExtension))
            elif mode=='flow':
                return os.path.join(inputPath, "low_%05d_%05d_flow%s" % (i, j, inputExtension))

        # Collect number of images
        num_images = 0
        num_frames = 0
        while True:
            if not os.path.exists(get_image_name(num_images, 0, 'low')):
                break
            num_images += 1
        while True:
            if not os.path.exists(get_image_name(0, num_frames, 'low')):
                break
            num_frames += 1
        print('Number of images found: %d, each with %d frames' % (num_images, num_frames))
        if number_of_images is not None and number_of_images>0:
            num_images = min(num_images, number_of_images)
            print('But limited to %d images'%number_of_images)

        # load all images
        #print('Load all images')
        pg = ProgressBar(num_images, 'Load all images', length=50)
        images_high = [None]*num_images
        images_low = [None]*num_images
        flow_low = [None]*num_images
        output_channels = 3
        for i in range(num_images):
            pg.print_progress_bar(i)
            high = [None]*num_frames
            low = [None]*num_frames
            flow = [None]*num_frames
            for j in range(num_frames):
                if not deferred_shading:
                    high[j] = np.clip(np.asarray(imageio.imread(get_image_name(i, j, 'high'))).transpose((2, 0, 1)), 0, 1)
                else:
                    high_rgb = np.clip(np.asarray(imageio.imread(get_image_name(i, j, 'high'))).transpose((2, 0, 1)), 0, 1)
                    high_dn = np.asarray(imageio.imread(get_image_name(i, j, 'highdn'))).transpose((2, 0, 1))
                    high[j] = np.concatenate((high_rgb, high_dn), axis=0)

                low_rgb = np.clip(np.asarray(imageio.imread(get_image_name(i, j, 'low'))).transpose((2, 0, 1)), 0, 1)
                if use_input_depth or use_input_normal:
                    low_dn = np.asarray(imageio.imread(get_image_name(i, j, 'dn'))).transpose((2, 0, 1))
                    if use_input_depth and use_input_normal:
                        low[j] = np.concatenate((low_rgb, low_dn), axis=0)
                    elif use_input_depth: #not use_input_normal
                        low[j] = np.concatenate((low_rgb, low_dn[3:4,:,:]), axis=0)
                    elif use_input_normal: #not use_input_depth
                        low[j] = np.concatenate((low_rgb, low_dn[0:3,:,:]), axis=0)
                else:
                    low[j] = low_rgb
                flow_xy = imageio.imread(get_image_name(i, j, 'flow'))[:,:,0:2]
                flow_inpaint = np.stack((
                    cv.inpaint(flow_xy[:,:,0], np.uint8(low_rgb[3,:,:]==0), 3, cv.INPAINT_NS),
                    cv.inpaint(flow_xy[:,:,1], np.uint8(low_rgb[3,:,:]==0), 3, cv.INPAINT_NS)), axis=0)
                low[j][3,:,:] = low[j][3,:,:] * 2 - 1 # transform mask to [-1,1]
                high[j][3,:,:] = high[j][3,:,:] * 2 - 1
                if deferred_shading:
                    channel_mask = [3, 4, 5, 6, 7] # mask, normal x, y, z, depth
                    low[j] = low[j][channel_mask,:,:]
                    high[j] = high[j][channel_mask,:,:]
                flow[j] = flow_inpaint
            images_high[i] = np.stack(high, axis=0)
            images_low[i] = np.stack(low, axis=0)
            flow_low[i] = np.stack(flow, axis=0)
        pg.print_progress_bar(num_images)
        if deferred_shading:
            input_channels = 5
            output_channels = 5
        else:
            input_channels = 4
            if use_input_depth:
                input_channels += 1
            if use_input_normal:
                input_channels += 3

    # find crops

    def randomPointOnSphere():
        vec = np.random.randn(3)
        vec /= np.linalg.norm(vec)
        return vec;

    print('Find crops')
    fill_ratio = 0.5 * video_crop_size * video_crop_size # at least 50% of that crop has to be filled
    samples = [None]*number_of_samples
    sample = 0
    while sample < number_of_samples:
        while True:
            index = random.randint(0, num_images-1)
            w = images_low[index].shape[2]
            h = images_low[index].shape[3]
            x = random.randint(0, w - video_crop_size - 1)
            y = random.randint(0, h - video_crop_size - 1)
            # check if it is filled
            crop_mask1 = (images_low[index][0,0,x:x+video_crop_size,y:y+video_crop_size]
                        + images_low[index][0,1,x:x+video_crop_size,y:y+video_crop_size]
                        + images_low[index][0,2,x:x+video_crop_size,y:y+video_crop_size]) > 0
            crop_mask2 = (images_low[index][num_frames-1,0,x:x+video_crop_size,y:y+video_crop_size]
                        + images_low[index][num_frames-1,1,x:x+video_crop_size,y:y+video_crop_size]
                        + images_low[index][num_frames-1,2,x:x+video_crop_size,y:y+video_crop_size]) > 0
            if np.sum(crop_mask1) >= fill_ratio and np.sum(crop_mask2) >= fill_ratio:
                # we found our sample
                samples[sample] = Sample(
                    index=index, 
                    crop_low=(x,x+video_crop_size,y,y+video_crop_size),
                    crop_high=(upsampling*x,upsampling*(x+video_crop_size),upsampling*y,upsampling*(y+video_crop_size)),
                    augmentation=np.random.randint(MAX_AUGMENTATION_MODE),
                    ambient_color=np.array([random.uniform(0.05,0.2)]*3), # color + light only needed for deferred shading
                    diffuse_color=np.array([random.uniform(0.4,1.0)]*3),
                    light_direction=np.array([0,0,1]*3) if random.uniform(0,1)<0.5 else randomPointOnSphere()
                )
                #print(samples[sample])
                sample += 1
                break
    #sort samples by image index for proper sepearation between test and training
    samples.sort(key = lambda s: s.index)
    print('All samples collected')

    return DatasetData(samples=samples, 
                       images_high=images_high,
                       images_low=images_low, 
                       flow_low=flow_low, 
                       input_channels=input_channels,
                       output_channels=output_channels,
                       crop_size = video_crop_size,
                       num_frames = num_frames)

class DatasetFromSamples(data.Dataset):
    """
    Output of __getitem__ (in that order):
     - images_low: num_frames x input_channels x crop_size x crop_size
     - flow_low: num_frames x 2 x crop_size x crop_size
     - images_high: num_frames x output_channels x crop_size*upsampling x crop_size*upsampling
    """
    def __init__(self, 
                 dataset_data, #The selected samples
                 test, #True: test set, False: training set
                 test_fraction): #Fraction of images used for the test set
        super(DatasetFromSamples, self).__init__()

        self.samples = dataset_data.samples
        self.data = dataset_data
        self.num_images = len(self.samples)

        l = int(self.num_images * test_fraction)
        if test:
            self.index_offset = self.num_images - l
            self.num_images = l
        else:
            self.index_offset = 0
            self.num_images -= l

    def get_low_res_shape(self, channels):
        upscale_factor = 4
        return (channels, self.data.crop_size, self.data.crop_size)

    def get_high_res_shape(self):
        return (self.data.images_high[0].shape[1], self.data.crop_size*4, self.data.crop_size*4)

    def get(self, index, mode):
        if mode=='high':
            img = self.data.images_high[self.samples[index].index]
            img = img[:,:, self.samples[index].crop_high[0]:self.samples[index].crop_high[1], self.samples[index].crop_high[2]:self.samples[index].crop_high[3]]
            return img
            #return torch.from_numpy(img)
        elif mode=='low':
            img = self.data.images_low[self.samples[index].index]
            img = img[:,:, self.samples[index].crop_low[0]:self.samples[index].crop_low[1], self.samples[index].crop_low[2]:self.samples[index].crop_low[3]]
            return img
            #return torch.from_numpy(img)
        elif mode=='flow':
            img = self.data.flow_low[self.samples[index].index]
            img = img[:,:,self.samples[index].crop_low[0]:self.samples[index].crop_low[1], self.samples[index].crop_low[2]:self.samples[index].crop_low[3]]
            return img
            #return torch.from_numpy(img)

    def __getitem__(self, index):
        low, high, flow = data_augmentation(
            self.get(index, 'low'), 
            self.get(index, 'high'), 
            self.get(index, 'flow'),
            self.samples[index].augmentation)
        return torch.from_numpy(low), torch.from_numpy(flow), torch.from_numpy(high)
    def __len__(self):
        return self.num_images

class DatasetFromFullImages(data.Dataset):
    def __init__(self, 
                 dataset_data, #The images
                 num_images):
        super(DatasetFromFullImages, self).__init__()

        self.data = dataset_data
        self.num_images = num_images

    def __getitem__(self, index):
        return torch.from_numpy(self.data.images_low[index]), torch.from_numpy(self.data.flow_low[index])

    def __len__(self):
        return self.num_images
