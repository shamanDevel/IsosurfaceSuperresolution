import torch.utils.data as data

import os.path
import imageio
import collections
import random
import numpy as np
import torch
import cv2 as cv
from joblib import Parallel, delayed
import time
from console_progressbar import ProgressBar

crop_size = 32 # arbitrary chosen, low-res value
               # chosen so that the high-res patch is of size 128^2
video_crop_size = 64

# Load and argument samples
Sample = collections.namedtuple('Sample', 'index,crop_high,crop_low')

def collect_samples_clouds(upsampling, opt):
    number_of_samples = opt.samples
    number_of_images = opt.numberOfImages
    use_input_depth = opt.useInputDepth
    use_input_normal = opt.useInputNormal
    inputPath = '../../data/clouds/rendering/'
    inputExtension = '.exr'

    def get_image_name(i,mode):
        if mode=='high':
            return os.path.join(inputPath, "high_%05d%s" % (i, inputExtension))
        elif mode=='low':
            return os.path.join(inputPath, "low_%05d%s" % (i, inputExtension))
        elif mode=='dn':
            return os.path.join(inputPath, "low_%05d_depth%s" % (i, inputExtension))

    # Collect number of images
    num_images = 0
    while True:
        if not os.path.exists(get_image_name(num_images, 'low')):
            break
        num_images += 1
    print('Number of images found: %d'%num_images)
    if number_of_images is not None and number_of_images>0:
        num_images = min(num_images, number_of_images)
        print('But limited to %d images'%number_of_images)

    # load all images
    print('Load all images')
    # #serial
    # t0 = time.time()
    images_high = [None]*num_images
    images_low = [None]*num_images
    for i in range(num_images):
        images_high[i] = np.asarray(imageio.imread(get_image_name(i,'high'))).transpose((2, 0, 1))
        low_rgb = np.asarray(imageio.imread(get_image_name(i,'low'))).transpose((2, 0, 1))
        if use_input_depth or use_input_normal:
            low_dn = np.asarray(imageio.imread(get_image_name(i,'dn'))).transpose((2, 0, 1))
            if use_input_depth and use_input_normal:
                images_low[i] = np.concatenate((low_rgb, low_dn), axis=0)
            elif use_input_depth: #not use_input_normal
                images_low[i] = np.concatenate((low_rgb, low_dn[3:4,:,:]), axis=0)
            elif use_input_normal: #not use_input_depth
                images_low[i] = np.concatenate((low_rgb, low_dn[0:3,:,:]), axis=0)
        else:
            images_low[i] = low_rgb
    input_channels = 4
    if use_input_depth:
        input_channels += 1
    if use_input_normal:
        input_channels += 3

    # find crops
    print('Find crops')
    fill_ratio = 0.5 * crop_size * crop_size # at least 50% of that crop has to be filled
    samples = [None]*number_of_samples
    sample = 0
    while sample < number_of_samples:
        while True:
            index = random.randint(0, num_images-1)
            w = images_low[index].shape[1]
            h = images_low[index].shape[2]
            x = random.randint(0, w - crop_size - 1)
            y = random.randint(0, h - crop_size - 1)
            # check if it is filled
            crop_mask = (images_low[index][0,x:x+crop_size,y:y+crop_size] + images_low[index][1,x:x+crop_size,y:y+crop_size] + images_low[index][2,x:x+crop_size,y:y+crop_size]) > 0
            if np.sum(crop_mask) >= fill_ratio:
                # we found our sample
                samples[sample] = Sample(index=index, crop_low=(x,x+crop_size,y,y+crop_size), crop_high=(upsampling*x,upsampling*(x+crop_size),upsampling*y,upsampling*(y+crop_size)))
                #print(samples[sample])
                sample += 1
                break
    print('All samples collected')

    return samples,images_high, images_low, input_channels

def collect_samples_div2k(upsampling, opt):
    number_of_samples = opt.samples
    number_of_images = opt.numberOfImages
    use_input_depth = opt.useInputDepth
    use_input_normal = opt.useInputNormal

    if use_input_depth or use_input_normal:
        print('Div2k dataset does not contain depth and normal information, ignore it')

    #inputPath = '../../data/div2ksmall/'
    inputPath = '../../data/div2k/'

    def get_image_name(i,high):
        i += 1
        if high:
            return os.path.join(inputPath, "%04d.png" % i)
        else:
            return os.path.join(inputPath, "%04dx4.png" % i)

    # Collect number of images
    num_images = 0
    while True:
        if not os.path.exists(get_image_name(num_images, False)):
            break
        num_images += 1
    print('Number of images found: %d'%num_images)
    if number_of_images is not None and number_of_images>0:
        num_images = min(num_images, number_of_images)
        print('But limited to %d images'%number_of_images)

    # load all images
    print('Load all images')
    images_high = [None]*num_images
    images_low = [None]*num_images
    for i in range(num_images):
        images_high[i] = np.asarray(imageio.imread(get_image_name(i,True))).transpose((2, 0, 1))
        images_low[i] = np.asarray(imageio.imread(get_image_name(i,False))).transpose((2, 0, 1))
        images_high[i] = np.concatenate((images_high[i]/255.0, np.ones((1, images_high[i].shape[1], images_high[i].shape[2]))), axis=0).astype(np.float32)
        images_low[i] = np.concatenate((images_low[i]/255.0, np.ones((1, images_low[i].shape[1], images_low[i].shape[2]))), axis=0).astype(np.float32)

    # find crops
    print('Find crops')
    samples = [None]*number_of_samples
    sample = 0
    while sample < number_of_samples:
        index = random.randint(0, num_images-1)
        w = images_low[index].shape[1]
        h = images_low[index].shape[2]
        x = random.randint(0, w - crop_size - 1)
        y = random.randint(0, h - crop_size - 1)
        samples[sample] = Sample(index=index, crop_low=(x,x+crop_size,y,y+crop_size), crop_high=(upsampling*x,upsampling*(x+crop_size),upsampling*y,upsampling*(y+crop_size)))
        sample += 1
    print('All samples collected')

    return samples,images_high, images_low, 4

class DatasetFromSamples(data.Dataset):
    def __init__(self, 
                 samples,images_high,images_low, #The selected samples
                 test, #True: test set, False: training set
                 test_fraction=0.2): #Fraction of images used for the test set
        super(DatasetFromSamples, self).__init__()

        self.samples = samples
        self.images_high = images_high
        self.images_low = images_low

        self.num_images = len(samples)

        l = int(self.num_images * test_fraction)
        if test:
            self.index_offset = self.num_images - l
            self.num_images = l
        else:
            self.index_offset = 0
            self.num_images -= l

    def get_low_res_shape(self, input_channels):
        return (input_channels, crop_size, crop_size)

    def get_high_res_shape(self):
        return (3, crop_size*4, crop_size*4)

    def get(self, index, high):
        if high:
            img = self.images_high[self.samples[index].index]
            img = img[0:3, self.samples[index].crop_high[0]:self.samples[index].crop_high[1], self.samples[index].crop_high[2]:self.samples[index].crop_high[3]]
            return torch.from_numpy(img)
        else:
            img = self.images_low[self.samples[index].index]
            img = img[:, self.samples[index].crop_low[0]:self.samples[index].crop_low[1], self.samples[index].crop_low[2]:self.samples[index].crop_low[3]]
            mask = (img[0,:,:] + img[1,:,:] + img[2,:,:]) > 0
            img[3,:,:] = mask*2-1
            return torch.from_numpy(img)

    def __getitem__(self, index):
        return self.get(index, False), self.get(index, True)

    def __len__(self):
        return self.num_images

class DatasetFromFullImages(data.Dataset):
    def __init__(self, 
                 images_low, #The images
                 num_images):
        super(DatasetFromFullImages, self).__init__()

        self.images_low = images_low
        self.num_images = num_images

    def get(self, index, high):
        if high:
            img = self.images_high[self.samples[index].index]
            img = img[0:3, self.samples[index].crop_high[0]:self.samples[index].crop_high[1], self.samples[index].crop_high[2]:self.samples[index].crop_high[3]]
            return torch.from_numpy(img)
        else:
            img = self.images_low[self.samples[index].index]
            img = img[:, self.samples[index].crop_low[0]:self.samples[index].crop_low[1], self.samples[index].crop_low[2]:self.samples[index].crop_low[3]]
            mask = (img[0,:,:] + img[1,:,:] + img[2,:,:]) > 0
            img[3,:,:] = mask*2-1
            return torch.from_numpy(img)

    def __getitem__(self, index):
        img = self.images_low[index]
        mask = (img[0,:,:] + img[1,:,:] + img[2,:,:]) > 0
        img[3,:,:] = mask*2-1
        return torch.from_numpy(img)

    def __len__(self):
        return self.num_images
