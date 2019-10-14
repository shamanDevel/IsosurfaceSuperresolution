import torch
import numpy as np
import matplotlib.pyplot as plt

import datasetVideo
from utils import ScreenSpaceShading

opt = dict({'samples':4, 'numberOfImages':1})
dataset_data = datasetVideo.collect_samples_clouds_video(
        4, opt, deferred_shading=True)
test_full_set = datasetVideo.DatasetFromFullImages(dataset_data, 1)
input_data, input_flow = test_full_set[0]
print(input_data.shape)

device = input_data.device
shading = ScreenSpaceShading(device)
shading.fov(30)
shading.ambient_light_color(np.array([0.1,0.1,0.1]))
shading.diffuse_light_color(np.array([1.0, 1.0, 1.0]))
shading.specular_light_color(np.array([0.2, 0.2, 0.2]))
shading.specular_exponent(16)
shading.light_direction(np.array([0.2,0.2,1.0]))
shading.material_color(np.array([1.0, 0.3, 0.3]))

output_rgb = shading(input_data)

f, axarr = plt.subplots(1,3)
axarr[0].imshow(input_data.numpy()[0,0,:,:]*0.5+0.5) #mask
axarr[1].imshow((input_data[0,1:4,:,:]*0.5+0.5).numpy().transpose((1,2,0))) #normal
axarr[2].imshow(output_rgb[0,:,:,:].numpy().transpose((1,2,0))) #rgb
plt.show()
