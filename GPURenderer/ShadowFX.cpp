#include "pch.h"
#include "ShadowFX.h"

#include <random>
#include <openvdb/math/Vec3.h>
#include <openvdb/math/Mat3.h>

ShadowFX::ShadowFX()
{
}


ShadowFX::~ShadowFX()
{
}

static float smoothstep(float a, float b, float t)
{
	t = std::max(0.0f, std::min(1.0f, (t - a) / (b - a)));
	return t * t * (3.0f - 2.0f*t);
}

void ShadowFX::screenSpaceAmbientOcclusion(
	int width, int height, 
	const float* input, int inputEntryStride, int inputRowStride, int inputNormalXIndex, int inputNormalYIndex, int inputNormalZIndex, int inputDepthIndex,
	float* output, int outputEntryStride, int outputRowStride, int outputIndex, 
	int samples, float sampleRadius, float bias, int noiseSamples)
{
	//normal is in screen space, pointing in positive y direction

	typedef openvdb::math::Vec3<float> Vec3;
	typedef openvdb::math::Mat3<float> Mat3;

	//generate sample hemisphere
	std::uniform_real_distribution<float> randomFloats(0.0, 1.0);
	std::default_random_engine generator(42);
	std::vector<Vec3> ssaoKernel(samples);
	for (unsigned int i = 0; i < samples; ++i)
	{
		Vec3 sample(
			randomFloats(generator) * 2.0 - 1.0,
			randomFloats(generator),
			randomFloats(generator) * 2.0 - 1.0
		);
		sample.normalize();
		sample *= randomFloats(generator);
		float scale = (float)i / 64.0;
		scale = 0.1f + scale * scale * (1.0f - 0.1f);
		sample *= scale;
		ssaoKernel[i] = sample;
	}

	//and random kernel rotations
	std::vector<std::vector<Vec3>> ssaoNoise(noiseSamples);
	for (int x=0; x<noiseSamples; ++x)
	{
		ssaoNoise[x].resize(noiseSamples);
		for (unsigned int y = 0; y < noiseSamples; y++)
		{
			Vec3 noise(
				randomFloats(generator) * 2.0 - 1.0,
				0.0f,
				randomFloats(generator) * 2.0 - 1.0);
			ssaoNoise[x][y] = noise;
		}
	}

	//create ambient occlusion
	std::vector<float> ssaoBuffer(width * height);
#pragma omp parallel for
	for (int y=0; y<height; ++y) for (int x=0; x<width; ++x)
	{
		//fetch input
		float depth = input[inputDepthIndex + inputEntryStride * (x + inputRowStride * y)];
		if (depth==0 || depth==1)
		{
			output[outputIndex + outputEntryStride * (x + outputRowStride * y)] = 0;
			continue;
		}
		Vec3 normal(
			input[inputNormalXIndex + inputEntryStride * (x + inputRowStride * y)],
			input[inputNormalYIndex + inputEntryStride * (x + inputRowStride * y)],
			input[inputNormalZIndex + inputEntryStride * (x + inputRowStride * y)]
		);
		normal.normalize();

		//compute SSAO

		Vec3 fragPos(x/float(width)*2-1, y/float(height)*2-1, depth*2-1);
		Vec3 randomVec = ssaoNoise[x % noiseSamples][y % noiseSamples];
		Vec3 tangent = randomVec - normal * randomVec.dot(normal); tangent.normalize();
		Vec3 bitangent = normal.cross(tangent);
		Mat3 TBN(tangent, bitangent, normal, false);

		float ssao = 0;
		for(int i =0; i<samples; ++i)
		{
			//get sample position
			Vec3 sample = fragPos + TBN * ssaoKernel[i] * sampleRadius;
			int sampleX = int(round((sample.x() + 1)*0.5*width));
			int sampleY = int(round((sample.y() + 1)*0.5*height));
			sampleX = std::min(width - 1, std::max(0, sampleX));
			sampleY = std::min(height - 1, std::max(0, sampleY));
			float sampleDepth = sample.z()*0.5+0.5;
			float screenDepth = input[inputDepthIndex + inputEntryStride * (sampleX + inputRowStride * sampleY)];
			float rangeCheck = smoothstep(0.0, 1.0, 0.5 / abs(depth - screenDepth));
			ssao += (screenDepth > sampleDepth + bias ? 1.0f : 0.0f) * rangeCheck;
		}
		ssao = 1 - (ssao / samples);

		//write result
		ssaoBuffer[x + width * y] = ssao;
	}

	//Blur
	const int blurRadius = noiseSamples / 2;
#pragma omp parallel for
	for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x)
	{
		/*
		float result = 0.0f;
		for (int y2=-blurRadius; y2<blurRadius; ++y2)
			for (int x2=-blurRadius; x2<blurRadius; ++x2)
			{
				int x3 = std::max(0, std::min(width - 1, x + x2));
				int y3 = std::max(0, std::min(height - 1, y + y2));
				result += ssaoBuffer[x3 + width * y3];
			}
		result /= noiseSamples * noiseSamples;
		*/
		float result = ssaoBuffer[x + width * y];
		output[outputIndex + outputEntryStride * (x + outputRowStride * y)] = result;
	}
}
