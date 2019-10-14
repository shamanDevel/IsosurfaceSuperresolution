#include "pch.h"
#include "ExternalImporter.h"

#include <fstream>
#include <boost/algorithm/algorithm.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <experimental/filesystem>
#include <openvdb/tools/Dense.h>

static void printProgress(const std::string& prefix, float progress)
{
	int barWidth = 50;
	std::cout << prefix << " [";
	int pos = barWidth * progress;
	for (int i = 0; i < barWidth; ++i) {
		if (i < pos) std::cout << "=";
		else if (i == pos) std::cout << ">";
		else std::cout << " ";
	}
	std::cout << "] " << int(progress * 100.0) << " %\r";
	std::cout.flush();
	if (progress >= 1) std::cout << std::endl;
}

openvdb::FloatGrid::Ptr ExternalImporter::importRAW(
	const std::string& filename, int downsampling, float lowerThreshold)
{
	if (!boost::algorithm::ends_with(filename, ".dat"))
	{
		std::cout << "Filename does not point to the .dat file" << std::endl;
		return nullptr;
	}

	//read descriptor file
	std::ifstream file(filename);
	if (!file.is_open())
	{
		std::cout << "Unable to open file " << filename << std::endl;
		return nullptr;
	}
	std::string line;
	std::string objectFileName = "";
	size_t resolutionX = 0;
	size_t resolutionY = 0;
	size_t resolutionZ = 0;
	std::string datatype = "";
	const std::string DATATYPE_UCHAR = "UCHAR";
	const std::string DATATYPE_USHORT = "USHORT";
	const std::string DATATYPE_BYTE = "BYTE";
	while (std::getline(file, line))
	{
		if (line.empty()) continue;
		std::istringstream iss(line);
		std::string token;
		iss >> token;
		if (!iss) continue; //no token in the current line
		if (token == "ObjectFileName:")
			iss >> objectFileName;
		else if (token == "Resolution:")
			iss >> resolutionX >> resolutionY >> resolutionZ;
		else if (token == "Format:")
			iss >> datatype;
		if (!iss)
		{
			std::cout << "Unable to parse line with token " << token << std::endl;
			return nullptr;
		}
	}
	file.close();
	if (objectFileName.empty() || resolutionX == 0 || datatype.empty())
	{
		std::cout << "Descriptor file does not contain ObjectFileName, Resolution and Format" << std::endl;
		return nullptr;
	}
	if (!(datatype == DATATYPE_UCHAR || datatype == DATATYPE_USHORT || datatype == DATATYPE_BYTE))
	{
		std::cout << "Unknown format " << datatype << std::endl;
		return nullptr;
	}
	std::cout << "Descriptor file read"
		<< "\nObjectFileName: " << objectFileName
		<< "\nResolution: " << resolutionX << ", " << resolutionY << ", " << resolutionZ
		<< "\nFormat: " << datatype << std::endl;

	// read volume
	size_t bytesPerEntry = 0;
	if (datatype == DATATYPE_UCHAR) bytesPerEntry = 1;
	if (datatype == DATATYPE_BYTE) bytesPerEntry = 1;
	if (datatype == DATATYPE_USHORT) bytesPerEntry = 2;
	size_t bytesToRead = resolutionX * resolutionY * resolutionZ * bytesPerEntry;
	std::string bfilename = std::experimental::filesystem::path(filename).replace_filename(objectFileName).generic_string();

	std::cout << "Load " << bytesToRead << " bytes from " << bfilename << std::endl;
	std::ifstream bfile(bfilename, std::ifstream::binary | std::ifstream::ate);
	if (!bfile.is_open())
	{
		std::cout << "Unable to open file " << bfilename << std::endl;
		return nullptr;
	}
	size_t filesize = bfile.tellg();
	int headersize = static_cast<int>(filesize - static_cast<long long>(bytesToRead));
	if (headersize < 0)
	{
		std::cout << "File is too small, " << (-headersize) << " bytes missing" << std::endl;
		return nullptr;
	}
	std::cout << "Skipping header of length " << headersize << std::endl;
	bfile.seekg(std::ifstream::pos_type(headersize));

#if 1
	bytesToRead = resolutionX * resolutionY * downsampling * bytesPerEntry;
	std::vector<char> data(bytesToRead);

	//convert to dense grid and build vdb grid slice by slice
	//openvdb::CoordBBox gridDim(0, 0, 0, resolutionX - 1, resolutionY - 1, resolutionZ - 1);
	//typedef openvdb::tools::Dense<float, openvdb::tools::LayoutZYX> DenseT;
	typedef openvdb::tools::Dense<float, openvdb::tools::LayoutXYZ> DenseT;
	openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create();
	const int sliceSize = resolutionX * resolutionY / (downsampling*downsampling);

	for (int z=0; z<resolutionZ/ downsampling; ++z)
	{
		bfile.read(&data[0], bytesToRead);
		if (!bfile)
		{
			std::cout << "Loading data file failed" << std::endl;
			return nullptr;
		}

		if (z%10 == 0)
			printProgress("Convert", z / float(resolutionZ));
		openvdb::CoordBBox sliceDim(0, 0, z, 
			resolutionX / downsampling - 1, resolutionY / downsampling - 1, z);
		DenseT d(sliceDim);

		if (datatype == DATATYPE_UCHAR || datatype==DATATYPE_BYTE) {
			const unsigned char* raw = reinterpret_cast<unsigned char*>(data.data());
#pragma omp parallel for
			for (int y = 0; y < resolutionY / downsampling; ++y)
				for (int x = 0; x < resolutionX / downsampling; ++x)
			{
				float value = 0;
				for (int iz = 0; iz < downsampling; ++iz)
					for (int iy=0; iy<downsampling; ++iy)
						for (int ix = 0; ix < downsampling; ++ix)
				{
					long long idx2 = (ix + downsampling * x)
							+ resolutionX * ((iy + downsampling * y) + resolutionY * iz);
					float val = raw[idx2] / 255.0f;
					value += val;
				}
				long long idx = x + y * resolutionX;
				value /= downsampling*downsampling;
				if (value < lowerThreshold) value = 0;
				d.setValue(size_t(idx), value);
			}
		}
		else if (datatype == DATATYPE_USHORT) {
			const unsigned short* raw = reinterpret_cast<unsigned short*>(data.data());
#pragma omp parallel for
			for (int y = 0; y < resolutionY / downsampling; ++y)
				for (int x = 0; x < resolutionX / downsampling; ++x)
			{
				float value = 0;
				for (int iz = 0; iz < downsampling; ++iz)
					for (int iy=0; iy<downsampling; ++iy)
						for (int ix = 0; ix < downsampling; ++ix)
				{
					long long idx2 = (ix + downsampling * x)
							+ resolutionX * ((iy + downsampling * y) + resolutionY * iz);
					float val = raw[idx2] / 65535.0f;
					value += val;
				}
				long long idx = x + y * resolutionX/ downsampling;
				value /= downsampling*downsampling;
				if (value < lowerThreshold) value = 0;
				d.setValue(size_t(idx), value);
			}
		}

		openvdb::tools::copyFromDense(d, *grid, 0.001f, false);
	}
	printProgress("Convert", 1.0f);

#else
	std::vector<char> data(bytesToRead);
	bfile.read(&data[0], bytesToRead);
	if (!bfile)
	{
		std::cout << "Loading data file failed" << std::endl;
		return nullptr;
	}

	// convert to dense grid
	openvdb::CoordBBox gridDim(0, 0, 0, resolutionX - 1, resolutionY - 1, resolutionZ - 1);
	typedef openvdb::tools::Dense<float, openvdb::tools::LayoutXYZ> DenseT;
	DenseT d(gridDim);
	if (datatype == DATATYPE_UCHAR) {
		const unsigned char* raw = reinterpret_cast<unsigned char*>(data.data());
#pragma omp parallel for
		// I have no idea about the order in which the volume is stored.
		// Let's just try it out
		for (long long idx = 0; idx < resolutionX*resolutionY*resolutionZ; ++idx)
		{
			float value = raw[idx] / 255.0f;
			if (value < lowerThreshold) value = 0;
			d.setValue(size_t(idx), value);
		}
	}
	else if (datatype == DATATYPE_USHORT) {
		const unsigned short* raw = reinterpret_cast<unsigned short*>(data.data());
#pragma omp parallel for
		// I have no idea about the order in which the volume is stored.
		// Let's just try it out
		for (long long idx = 0; idx < resolutionX*resolutionY*resolutionZ; ++idx)
		{
			float value = raw[idx] / 65535.0f;
			if (value < lowerThreshold) value = 0;
			d.setValue(size_t(idx), value);
		}
	}

	//convert to vdb grid
	std::cout << "Build VDB grid" << std::endl;
	openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create();
	openvdb::tools::copyFromDense(d, *grid, 0.001f, false);
#endif

	//done
	std::cout << "Done!" << std::endl;
	return grid;
}
