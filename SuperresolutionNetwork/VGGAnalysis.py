"""
Computes the mean, variance, min and max for each layer in the VGG network
"""

import argparse
import math
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from console_progressbar import ProgressBar

import datasetVideo
import losses

parser = argparse.ArgumentParser(description='VGG Layer Response analysis')

parser.add_argument('--dataset', type=str, default='cloud-video', help="only 'cloud-video' supported")
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--numberOfImages', type=int, default=-1, help="Number of images taken from the inpt dataset. Default: -1 = unlimited")
parser.add_argument('--useInputDepth', action='store_true', help="Use depth information from the input images?")
parser.add_argument('--useInputNormal', action='store_true', help="Use normal information from the input images?")
parser.add_argument('--samples', type=int, default=1000, help='Number of samples for the train and test dataset')
parser.add_argument('--batchSize', type=int, default=64, help='training batch size')

opt = parser.parse_args()
device = torch.device("cpu")

# load samples and create dataset
dataset_data = datasetVideo.collect_samples_clouds_video(opt.upscale_factor, opt)
class HighResDatasetFromSamples(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()

        self.samples = dataset_data.samples
        self.data = dataset_data
        self.num_images = len(self.samples)

    def get_high_res_shape(self):
        return (self.data.output_channels, self.data.crop_size*4, self.data.crop_size*4)

    def __getitem__(self, index):
        img = self.data.images_high[self.samples[index].index]
        img = img[:,:, self.samples[index].crop_high[0]:self.samples[index].crop_high[1], self.samples[index].crop_high[2]:self.samples[index].crop_high[3]]
        return torch.from_numpy(img)

    def __len__(self):
        return self.num_images
test_set = HighResDatasetFromSamples()
testing_data_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=opt.batchSize, shuffle=True)
# build VGG
cnn = torchvision.models.vgg19(pretrained=True).features.eval()
means = defaultdict(float)
sqmeans = defaultdict(float)
variances = defaultdict(float)
minima = defaultdict(float)
maxima = defaultdict(float)
keys = []
class LayerStatistics(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
    def forward(self, x):
        means[self.name] += float(torch.mean(x))
        sqmeans[self.name] += float(torch.mean(x**2))
        variances[self.name] += float(torch.var(x))
        minima[self.name] = min(minima[self.name], float(torch.min(x)))
        maxima[self.name] = max(maxima[self.name], float(torch.max(x)))
        return x
model = nn.Sequential(losses.LossBuilder.VGGNormalization(device))
i = 0  # increment every time we see a conv
for layer in cnn.children():
    if isinstance(layer, nn.Conv2d):
        i += 1
        name = 'conv_{}'.format(i)
        model.add_module(name, layer)
        model.add_module(name+'_stats', LayerStatistics(name))
        keys.append(name)
    elif isinstance(layer, nn.ReLU):
        name = 'relu_{}'.format(i)
        layer = nn.ReLU(inplace=False)
        model.add_module(name, layer)
    elif isinstance(layer, nn.MaxPool2d):
        name = 'pool_{}'.format(i)
        model.add_module(name, layer)
    elif isinstance(layer, nn.BatchNorm2d):
        name = 'bn_{}'.format(i)
        model.add_module(name, layer)

# run and collect the statistics
with torch.no_grad():
    num_minibatch = len(testing_data_loader)
    pg = ProgressBar(num_minibatch, 'Testing', length=50)
    for iteration, batch in enumerate(testing_data_loader, 0):
        pg.print_progress_bar(iteration)
        batch = batch.to(device)
        for j in range(dataset_data.num_frames):
            img = batch[:,j,:,:,:]
            model(img)
    pg.print_progress_bar(num_minibatch)
    for key in keys:
        means[key] /= num_minibatch
        sqmeans[key] /= num_minibatch
        variances[key] /= num_minibatch

#different variance computation
variances2 = {}
for key in keys:
    variances2[key] = sqmeans[key] - means[key]**2

print('means:', means)
print('sqmeans:',sqmeans)
print('var1:',variances)
print('var2:',variances2)

# print results
print("   KEYS        MEAN          VAR          MIN           MAX")
for key in keys:
    print("%10s   % 8.3e   % 8.3e   % 8.3e   % 8.3e"%(key,means[key],variances2[key],minima[key],maxima[key]))

print()
print("Weights based on the variance:")
weights = []
for key in keys:
    weights.append(key+":%f"%(1.0/variances2[key]))
print(','.join(weights))

print()
print("Weights based on min-max:")
weights = []
for key in keys:
    weights.append(key+":%f"%(1.0/(maxima[key]-minima[key])))
print(','.join(weights))