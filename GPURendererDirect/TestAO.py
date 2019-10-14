import numpy as np
from math import sqrt, cos, sin

def _singleHemisphereSample():
    u1 = np.random.random()
    u2 = np.random.random()
    r = sqrt(u1);
    theta = 2 * np.pi * u2;
    x = r * cos(theta);
    y = r * sin(theta);
    vec = np.array([x, y, sqrt(1.0-u1)])
    scale = np.random.random()
    scale = 0.1 + 0.9 * scale * scale
    return vec * scale

def sampleHemisphere(normal, numSamples=16):
    normal = normal / np.linalg.norm(normal)
    print('normal:', normal)

    # random rotation vector
    noise = np.array([np.random.random()*2-1,
                      np.random.random()*2-1,
                      0])
    noise /= np.linalg.norm(noise)

    # transformation
    tangent = noise - normal * np.dot(noise, normal)
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(normal, tangent)
    print('tangent:', tangent)
    print('bitangent:', bitangent)
    TBN = np.column_stack((tangent, bitangent, normal))

    # generate samples
    samples = [None] * numSamples
    for i in range(numSamples):
        sampleT = _singleHemisphereSample() #tangent space
        sampleN = np.matmul(TBN, sampleT) #world space
        print('sample tangent-space:', sampleT, ', world-space:', sampleN)
        samples[i] = sampleN
    return samples

# Test
sampleHemisphere(np.array([0,1,0]))
sampleHemisphere(np.array([0,-1,0]))
sampleHemisphere(np.array([1,0,0]))
sampleHemisphere(np.array([0,0,1]))
