import os
import os.path
import random
import numpy as np
import numpy.linalg
import subprocess
import imageio

########################################
# CONFIGURATION
########################################
renderer = 'CPURenderer.exe'
datasetPath = '../../data/clouds/input/cloud-049.vdb'
datasetExtensions = tuple(['.vdb'])
outputPath = 'pipetest_%03d_%s'
numFrames = 10
maxDist = 0.3
resolutionX = 256
resolutionY = 128

########################################
# MAIN
########################################
def randomPointOnSphere():
    vec = np.random.randn(3)
    vec /= np.linalg.norm(vec)
    return vec;

def randomFloat(min, max):
    return min + np.random.random() * (max-min)

def interpolate(x, y, a):
    return (1-a)*x + a*y

def main():
    #initial parameters
    originStart = randomPointOnSphere() * randomFloat(1.0, 1.2)
    lookAtStart = randomPointOnSphere() * 0.1
    while True:
        originEnd = randomPointOnSphere() * randomFloat(1.0, 1.2)
        if numpy.linalg.norm(originEnd - originStart) < maxDist:
            break
    lookAtEnd = randomPointOnSphere() * 0.1
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

    #launch renderer
    args = [
        renderer,
        '-m','iso',
        '--res', '%d,%d'%(resolutionX,resolutionY),
        '--origin', '%5.3f,%5.3f,%5.3f'%(originStart[0],originStart[1],originStart[2]),
        '--lookat', '%5.3f,%5.3f,%5.3f'%(lookAtStart[0],lookAtStart[1],lookAtStart[2]),
        '--up', '%5.3f,%5.3f,%5.3f'%(up[0],up[1],up[2]),
        '--isovalue', str(isovalue),
        '--diffuse', '%5.3f,%5.3f,%5.3f'%(diffuseColor[0],diffuseColor[1],diffuseColor[2]),
        '--specular', '%5.3f,%5.3f,%5.3f'%(specularColor[0],specularColor[1],specularColor[2]),
        '--exponent', str(specularExponent),
        '--light', light,
        datasetPath,
        'PIPE'
        ]
    print(' '.join(args))
    sp = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    #loop over all frames
    numitems = 10 * resolutionY * resolutionX
    image = np.zeros((10, resolutionY, resolutionX), dtype=np.float32)
    for i in range(numFrames):
        currentOrigin = interpolate(originStart, originEnd, i / (numFrames-1.0))
        currentLookAt = interpolate(lookAtStart, lookAtEnd, i / (numFrames-1.0))
        sp.stdin.write(("cameraOrigin=%5.3f,%5.3f,%5.3f\n"%(currentOrigin[0], currentOrigin[1], currentOrigin[2])).encode("ascii"))
        sp.stdin.flush()
        sp.stdin.write(("cameraLookAt=%5.3f,%5.3f,%5.3f\n"%(currentLookAt[0], currentLookAt[1], currentLookAt[2])).encode("ascii"))
        sp.stdin.flush()
        sp.stdin.write(b"render\n")
        sp.stdin.flush()
        print("Rendering triggered, let's wait for the result")
        imagedata = sp.stdout.read(numitems * 4)
        image = np.frombuffer(imagedata, dtype=np.float32, count=numitems)
        #image = np.fromfile(sp.stdout, dtype=np.float32, count=numitems)
        image = np.reshape(image, (10, resolutionY, resolutionX))
        #for c in range(10):
        #    for y in range(resolutionY):
        #        imagedata = sp.stdout.read(resolutionX * 4)
        #        image[c,y,:] = np.frombuffer(imagedata, dtype=np.int32, count=resolutionX)# / 1000.0
        #        #image[c,y,:] = np.fromfile(sp.stdout, dtype=np.float32, count=resolutionX)
        imageio.imwrite(outputPath%(i,'rgb.png'), image[0:3,:,:].transpose((1, 2, 0)))
        imageio.imwrite(outputPath%(i,'mask.png'), image[3,:,:])
        imageio.imwrite(outputPath%(i,'normal.png'), (image[4:7,:,:]*0.5 + 0.5).transpose((1, 2, 0)))
        imageio.imwrite(outputPath%(i,'depth.exr'), image[7,:,:])
        imageio.imwrite(outputPath%(i,'flow.png'), 
                        np.concatenate([image[8:10,:,:]*10 + 0.5, np.zeros((1, resolutionY, resolutionX))], axis=0)
                        .transpose((1, 2, 0)))

    #kill renderer
    sp.stdin.write(b"exit\n")
    sp.stdin.flush()
    print('exit signal written')
    sp.wait(5)

if __name__ == "__main__":
    main()