import os
import os.path
import random
import numpy as np
import numpy.linalg
import subprocess
import imageio
import cv2 as cv

########################################
# CONFIGURATION
########################################
renderer = '../bin/GPURenderer.exe'
datasetPath = '../../data/volumes/vbx/'
descriptorFile = '../../data/volumes/inputs.dat'
#datasetExtensions = tuple(['.vbx'])
outputPath = '../../data/clouds/rendering_video5/'
outputExtension = '.exr'
numImages = 50
numFrames = 10
downscaling = 4
highResSize = 512
samplesHigh = 8
maxDist = 0.3
noShading = True
aoSamples = 256
aoRadius = 1.0

########################################
# MAIN
########################################
def randomPointOnSphere():
    vec = np.random.randn(3)
    vec /= np.linalg.norm(vec)
    vec[2] = - abs(vec[2])
    return vec;

def randomFloat(min, max):
    return min + np.random.random() * (max-min)

def convertToNumpy(inputPath, idx):
    # copied from datasetVideo.py
    inputExtension = ".exr"
    def get_image_name(i,j,mode):
        if mode=='high':
            return os.path.join(inputPath, "high_tmp_%05d%s" % (j, inputExtension))
        if mode=='highdn':
            return os.path.join(inputPath, "high_tmp_%05d_depth%s" % (j, inputExtension))
        if mode=='highfx':
            return os.path.join(inputPath, "high_tmp_%05d_fx%s" % (j, inputExtension))
        elif mode=='low':
            return os.path.join(inputPath, "low_tmp_%05d%s" % (j, inputExtension))
        elif mode=='dn':
            return os.path.join(inputPath, "low_tmp_%05d_depth%s" % (j, inputExtension))
        elif mode=='flow':
            return os.path.join(inputPath, "low_tmp_%05d_flow%s" % (j, inputExtension))
    high = [None]*numFrames
    low = [None]*numFrames
    flow = [None]*numFrames
    for j in range(numFrames):
        high_rgb = np.clip(np.asarray(imageio.imread(get_image_name(idx, j, 'high'))).transpose((2, 0, 1)), 0, 1)
        high_dn = np.asarray(imageio.imread(get_image_name(idx, j, 'highdn'))).transpose((2, 0, 1))
        high_fx = np.asarray(imageio.imread(get_image_name(idx, j, 'highfx'))).transpose((2, 0, 1))
        high[j] = np.concatenate((high_rgb[3:4,:,:], high_dn, high_fx[0:1,:,:]), axis=0)
        high[j][0,:,:] = high[j][0,:,:] * 2 - 1
        assert high[j].shape[0]==6

        low_rgb = np.clip(np.asarray(imageio.imread(get_image_name(idx, j, 'low'))).transpose((2, 0, 1)), 0, 1)
        low_dn = np.asarray(imageio.imread(get_image_name(idx, j, 'dn'))).transpose((2, 0, 1))
        low[j] = np.concatenate((low_rgb[3:4], low_dn), axis=0)
        low[j][0,:,:] = low[j][0,:,:] * 2 - 1 # transform mask to [-1,1]
        assert low[j].shape[0]==5

        flow_xy = imageio.imread(get_image_name(idx, j, 'flow'))[:,:,0:2]
        flow_inpaint = np.stack((
            cv.inpaint(flow_xy[:,:,0], np.uint8(low_rgb[3,:,:]==0), 3, cv.INPAINT_NS),
            cv.inpaint(flow_xy[:,:,1], np.uint8(low_rgb[3,:,:]==0), 3, cv.INPAINT_NS)), axis=0)
        flow[j] = flow_inpaint
    images_high = np.stack(high, axis=0)
    images_low = np.stack(low, axis=0)
    flow_low = np.stack(flow, axis=0)
    # save as numpy array
    np.save(os.path.join(inputPath, "high_%05d.npy" % idx), images_high)
    np.save(os.path.join(inputPath, "low_%05d.npy" % idx), images_low)
    np.save(os.path.join(inputPath, "flow_%05d.npy" % idx), flow_low)

def main():

    #create output
    if not os.path.exists(outputPath):
        os.mkdir(outputPath)

    #list all datasets
    dataset_info = np.genfromtxt(descriptorFile, skip_header=1, dtype=None)
    num_files = dataset_info.shape[0]
    datasets = [None]*num_files
    print('Datasets:')
    for i in range(num_files):
        name = str(dataset_info[i][0].decode('ascii'))
        min_iso = float(dataset_info[i][1])
        max_iso = float(dataset_info[i][2])
        datasets[i] = (name, min_iso, max_iso)
        print(name,"  iso=[%f,%f]"%(min_iso, max_iso))

    ##list all datasets
    #datasets = [file for file in os.listdir(datasetPath) if file.endswith(datasetExtensions)]
    #print('Datasets found:', datasets)

    #render images
    for i in range(numImages):
        print('Generate file',(i+1),'of',numImages)
        set = random.randrange(num_files)
        inputFile = os.path.join(datasetPath, datasets[set][0])
        outputFileHigh = os.path.join(outputPath, "high_tmp_&05d%s" % (outputExtension))
        outputFileDepthHigh = os.path.join(outputPath, "high_tmp_&05d_depth%s" % (outputExtension))
        outputFileEffectsHigh = os.path.join(outputPath, "high_tmp_&05d_fx%s" % (outputExtension))
        outputFileLow = os.path.join(outputPath, "low_tmp_&05d%s" % (outputExtension))
        outputFileDepthLow = os.path.join(outputPath, "low_tmp_&05d_depth%s" % (outputExtension))
        outputFileFlowLow = os.path.join(outputPath, "low_tmp_&05d_flow%s" % (outputExtension))
        originStart = randomPointOnSphere() * randomFloat(0.6, 1.0) + np.array([0,0,-0.07])
        lookAtStart = randomPointOnSphere() * 0.1  + np.array([0,0,-0.07])
        while True:
            originEnd = randomPointOnSphere() * randomFloat(0.6, 1.0) + np.array([0,0,-0.07])
            if numpy.linalg.norm(originEnd - originStart) < maxDist:
                break
        lookAtEnd = randomPointOnSphere() * 0.1 + np.array([0,0,-0.07])
        up = np.array([0,0,-1])#randomPointOnSphere()
        isovalue = random.uniform(datasets[set][1], datasets[set][2])
        diffuseColor = np.random.uniform(0.2,1.0,3)
        specularColor = [pow(random.uniform(0,1), 3)*0.3] * 3
        specularExponent = random.randint(4, 64)
        if random.uniform(0,1)<0.7:
            light =  'camera'
        else:
            lightDir = randomPointOnSphere()
            light = '%5.3f,%5.3f,%5.3f'%(lightDir[0],lightDir[1],lightDir[2])

        args = [
            renderer,
            '-m','iso',
            '--res', '%d,%d'%(highResSize,highResSize),
            '--animation', '%d'%numFrames,
            '--origin', '%5.3f,%5.3f,%5.3f,%5.3f,%5.3f,%5.3f'%(originStart[0],originStart[1],originStart[2],originEnd[0],originEnd[1],originEnd[2]),
            '--lookat', '%5.3f,%5.3f,%5.3f,%5.3f,%5.3f,%5.3f'%(lookAtStart[0],lookAtStart[1],lookAtStart[2],lookAtEnd[0],lookAtEnd[1],lookAtEnd[2]),
            '--up', '%5.3f,%5.3f,%5.3f'%(up[0],up[1],up[2]),
            '--isovalue', str(isovalue),
            '--noshading', '1' if noShading else '0',
            '--diffuse', '%5.3f,%5.3f,%5.3f'%(diffuseColor[0],diffuseColor[1],diffuseColor[2]),
            '--specular', '%5.3f,%5.3f,%5.3f'%(specularColor[0],specularColor[1],specularColor[2]),
            '--exponent', str(specularExponent),
            '--light', light,
            '--samples', str(samplesHigh),
            '--downscale_path', outputFileLow,
            '--downscale_factor', str(downscaling),
            '--depth', outputFileDepthLow,
            '--flow', outputFileFlowLow,
            '--highdepth', outputFileDepthHigh,
            '--ao', 'world',
            '--aosamples', str(aoSamples),
            '--aoradius', str(aoRadius),
            '--higheffects', outputFileEffectsHigh,
            inputFile,
            outputFileHigh
            ]
        print(' '.join(args))
        subprocess.run(args, stdout=None, stderr=None, check=True)

        print('Convert to Numpy')
        convertToNumpy(outputPath, i)

        # clean up
        for i in range(numFrames):
            os.remove((outputFileHigh.replace('&','%'))%i)
            os.remove((outputFileDepthHigh.replace('&','%'))%i)
            os.remove((outputFileEffectsHigh.replace('&','%'))%i)
            os.remove((outputFileLow.replace('&','%'))%i)
            os.remove((outputFileDepthLow.replace('&','%'))%i)
            os.remove((outputFileFlowLow.replace('&','%'))%i)

if __name__ == "__main__":
    main()