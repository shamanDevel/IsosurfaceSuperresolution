#pragma once

#include <string>
#include <openvdb/openvdb.h>

class ExternalImporter
{
private:
	ExternalImporter() {};

public:
	/**
	 * Imports the RAW-file described by the specified .dat file
	 * \param filename the .dat file
	 * \param lowerThreshold lower clipping for the density
	 */
	static openvdb::FloatGrid::Ptr importRAW(
		const std::string& filename,
		int downsampling = 1,
		float lowerThreshold = 0.01);
};

