#pragma once

class ShadowFX
{
public:
	ShadowFX();
	~ShadowFX();

	/**
	 * \brief Computes ambient occlusion in screen space.
	 * 
	 * The input array is indexed as following:
	 * normal.x = input[inputNormalXIndex + inputEntryStride*(x + inputRowStride*y)]
	 * 
	 * \param width 
	 * \param height 
	 * \param input 
	 * \param inputEntryStride 
	 * \param inputRowStride 
	 * \param inputNormalXIndex 
	 * \param inputNormalYIndex 
	 * \param inputNormalZIndex 
	 * \param inputDepthIndex 
	 * \param output 
	 * \param outputEntryStride 
	 * \param outputRowStride 
	 * \param outputIndex 
	 * \param samples 
	 * \param noiseSamples 
	 */
	static void screenSpaceAmbientOcclusion(
		int width, int height,
		const float* input, int inputEntryStride, int inputRowStride,
		int inputNormalXIndex, int inputNormalYIndex, int inputNormalZIndex, int inputDepthIndex,
		float* output, int outputEntryStride, int outputRowStride, int outputIndex,
		int samples = 64,
		float sampleRadius = 0.1, //screen space
		float bias = 0.025f,
		int noiseSamples = 4
	);
};

