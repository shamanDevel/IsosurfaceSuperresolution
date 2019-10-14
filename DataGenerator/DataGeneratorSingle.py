import os
import os.path
import random
import numpy as np
import subprocess

########################################
# CONFIGURATION
########################################
renderer = 'CPURenderer.exe'
datasetPath = '../../data/clouds/input/'
datasetExtensions = tuple(['.vdb'])
outputPath = '../../data/clouds/rendering/'
outputExtension = '.exr'
numImages = 500
downscaling = 4
highResSize = 512
samplesHigh = 8

########################################
# MAIN
########################################
def randomPointOnSphere():
    vec = np.random.randn(3)
    vec /= np.linalg.norm(vec)
    return vec;

def main():
    #create output
    if not os.path.exists(outputPath):
        os.mkdir(outputPath)

    #list all datasets
    datasets = [file for file in os.listdir(datasetPath) if file.endswith(datasetExtensions)]
    print('Datasets found:', datasets)

    #render images
    for i in range(numImages):
        print('Generate file',(i+1),'of',numImages)
        inputFile = os.path.join(datasetPath, random.choice(datasets))
        outputFileHigh = os.path.join(outputPath, "high_%05d%s" % (i, outputExtension))
        outputFileLow = os.path.join(outputPath, "low_%05d%s" % (i, outputExtension))
        outputFileDepthLow = os.path.join(outputPath, "low_%05d_depth%s" % (i, outputExtension))
        origin = randomPointOnSphere() * 1.0
        up = randomPointOnSphere()
        isovalue = random.uniform(0.1, 0.5)
        diffuseColor = np.random.uniform(0.2,1.0,3)
        specularColor = [pow(random.uniform(0,1), 3)*0.3] * 3
        specularExponent = random.randint(4, 64)
        if random.uniform(0,1)<0.5:
            light =  'camera'
        else:
            lightDir = randomPointOnSphere()
            light = '%5.3f,%5.3f,%5.3f'%(lightDir[0],lightDir[1],lightDir[2])

        args = [
            renderer,
            '-m','iso',
            '--res', '%d,%d'%(highResSize,highResSize),
            '--origin', '%5.3f,%5.3f,%5.3f'%(origin[0],origin[1],origin[2]),
            '--up', '%5.3f,%5.3f,%5.3f'%(up[0],up[1],up[2]),
            '--isovalue', str(isovalue),
            '--diffuse', '%5.3f,%5.3f,%5.3f'%(diffuseColor[0],diffuseColor[1],diffuseColor[2]),
            '--specular', '%5.3f,%5.3f,%5.3f'%(specularColor[0],specularColor[1],specularColor[2]),
            '--exponent', str(specularExponent),
            '--light', light,
            '--samples', str(samplesHigh),
            '--downscale_path', outputFileLow,
            '--downscale_factor', str(downscaling),
            '--depth', outputFileDepthLow,
            inputFile,
            outputFileHigh
            ]
        print(' '.join(args))
        subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

if __name__ == "__main__":
    main()