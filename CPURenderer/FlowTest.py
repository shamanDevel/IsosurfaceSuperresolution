import numpy as np
import cv2 as cv
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F

inputName = "../bin/outlow_000.exr"
outputName1 = "../bin/outlow_001_warpedNumpy.exr"
outputName2 = "../bin/outlow_001_warpedTorch.exr"
flowName = "../bin/outlowf_000.exr"
flowTest1 = "../bin/outlowf_000_t1.png"
flowTest2 = "../bin/outlowf_000_t2.png"

def warp_flow(img, flow): # apply the flowo to warp the img
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] = flow[:,:,0]*w + np.arange(w)
    flow[:,:,1] = flow[:,:,1]*h + np.arange(h)[:,np.newaxis]
    ih, iw = img.shape[:2]
    
    flow[:,:,0] = np.clip(flow[:,:,0], 0, iw)
    flow[:,:,1] = np.clip(flow[:,:,1], 0, ih)
    res = cv.remap(img, np.float32(flow), None, cv.INTER_LINEAR )
    return res
	
def warp_flow_torch(img, flow):
	H, W = flow.shape[:2]
	
	grid_offsetsH = torch.linspace(-1, +1, H)
	grid_offsetsW = torch.linspace(-1, +1, W)
	grid_offsetsH = torch.unsqueeze(grid_offsetsH, 1)
	grid_offsetsW = torch.unsqueeze(grid_offsetsW, 0)
	grid_offsets = torch.stack(
		torch.broadcast_tensors(grid_offsetsW, grid_offsetsH), 
		dim=2)
	grid_offsets = torch.unsqueeze(grid_offsets, 0) # batch dimension
	
	img_torch = torch.from_numpy(img.transpose((2, 0, 1)))
	img_torch = torch.unsqueeze(img_torch, 0)
	
	flow_torch = torch.unsqueeze(torch.from_numpy(flow), 0)
	
	grid = grid_offsets + flow_torch * -2
	warped = F.grid_sample(img_torch, grid)
	
	res = warped[0].numpy().transpose((1, 2, 0))
	return res
	
inputImage = imageio.imread(inputName)
flowImage = imageio.imread(flowName)[:,:,0:2]
print('inputImage:', inputImage.shape)
print('flowImage:', flowImage.shape)
imageio.imwrite(flowTest1, np.concatenate((flowImage + 0.5, np.zeros((flowImage.shape[0], flowImage.shape[1], 1))), axis=2))

imageio.imwrite("../bin/outlowf_000_t3.png", inputImage[:,:,3])

print()
print(inputImage[:,:,3])
print()
print(flowImage[:,:,0])
print()
print(np.uint8(inputImage[:,:,3]==0))
print()

# VERY IMPORTANT!!
flowImageInpainted = np.stack((
	cv.inpaint(flowImage[:,:,0], np.uint8(inputImage[:,:,3]==0), 3, cv.INPAINT_NS),
	cv.inpaint(flowImage[:,:,1], np.uint8(inputImage[:,:,3]==0), 3, cv.INPAINT_NS)), axis=2)
imageio.imwrite(flowTest2, np.concatenate((flowImageInpainted + 0.5, np.zeros((flowImage.shape[0], flowImage.shape[1], 1))), axis=2))



print(flowImageInpainted[:,:,0])
print()

warpedImage1 = warp_flow(inputImage, flowImageInpainted)
imageio.imwrite(outputName1, warpedImage1)

warpedImage2 = warp_flow_torch(inputImage, flowImageInpainted)
imageio.imwrite(outputName2, warpedImage2)