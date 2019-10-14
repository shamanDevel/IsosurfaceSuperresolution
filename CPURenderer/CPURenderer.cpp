// CPURenderer.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include <iostream>
#include <sstream>
#include <array>
#include <algorithm>
#include <chrono>
#include <CLI11.hpp>
#include <tinyformat.h>
#include <openvdb/tools/RayTracer.h>
#include <openvdb/io/Stream.h>
#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfFrameBuffer.h>
#include <OpenEXR/ImfHeader.h>
#include <OpenEXR/ImfOutputFile.h>
#include <OpenEXR/ImfPixelType.h>
#include <boost/algorithm/string/predicate.hpp>

#include <fcntl.h>
#include <io.h>

#include "PhongShader.h"
#include "IsoVolumeRayTracer.h"
#include "../CPURenderer/ExternalImporter.h"

template<typename Arg>
void parseStrImpl(const std::string& str, std::size_t pos, Arg& arg)
{
	if (str.find(',',pos) != std::string::npos)
		throw std::exception("more list entries specified than expected");
	std::stringstream ss(str.substr(pos));
	ss >> arg;
}

template<typename Arg, typename... Rest>
void parseStrImpl(const std::string& str, std::size_t pos, Arg& arg, Rest& ... rest)
{
	size_t p = str.find(',', pos);
	if (p == std::string::npos)
		throw std::exception("less list entries specified than expected");
	std::stringstream ss(str.substr(pos, p));
	ss >> arg;
	parseStrImpl(str, p + 1, rest...);
}

template<typename ...Args>
void parseStr(const std::string& str, Args & ... args)
{
	parseStrImpl(str, 0, args...);
}

//Does the string contain a '%d' or similiar?
bool hasIntFormat(const std::string& str)
{
	try
	{
		int index = 1;
		tfm::format(str, index);
		return true;
	} catch (std::exception& ex)
	{
		return false;
	}
}

struct Args
{
	//input/output
	std::string inputFilename;
	std::string outputFilename;
	//downscale
	int downscaleFactor = 1;
	std::string downscaleFilename;
	std::string downscaleDepthFilename;
	std::string highDepthFilename;
	//animation
	int animFrames = 0;
	std::string downscaleFlowFilename;
	openvdb::Vec3d cameraOriginEnd = { 0,0,-1 };
	openvdb::Vec3d cameraLookAtEnd = { 0,0,0 };
	//camera
	int resolutionX = 512;
	int resolutionY = 512;
	double cameraFov = 45;
	openvdb::Vec3d cameraOrigin = {0,0,-1};
	openvdb::Vec3d cameraLookAt = {0,0,0};
	openvdb::Vec3d cameraUp = {0,1,0};
	//render mode
	static const std::string RenderModeVolume;
	static const std::string RenderModeIso;
	static const std::string RenderModeConvert;
	std::string renderMode;
	int noShading = 0;
	//isosurface
	double isovalue = 0.0;
	openvdb::Vec3d materialDiffuse = { 0.7, 0.7, 0.7 };
	openvdb::Vec3d materialSpecular = { 1,1,1 };
	openvdb::Vec3d materialAmbient = { 0.1,0.1,0.1 };
	int materialSpecularExponent = 32;
	bool cameraLight = true; //true: light along camera view direction
	openvdb::Vec3d lightDirection = {0,0,0};
	int samples = 1;
	//iso-effects (unused)
	std::string aoMode = "";
	int aoSamples = 32;
	float aoRadius = 0.01;
	std::string highEffectsFilename;
	//volume
	//TODO
	//pipe
	std::string pipeOutputFormat = "RGBMXYZDUV"; //red, green, blue, mask, normal x, normal y, normal z, depth, flow u, flow v
};
const std::string Args::RenderModeVolume = "volume";
const std::string Args::RenderModeIso = "iso";
const std::string Args::RenderModeConvert = "convert";

Args parseArguments(int argc, const char* argv[])
{
	CLI::App app{ "CPU Volume Renderer" };
	Args args;

	//input and output
	app.add_option("input,-i,--input", args.inputFilename, 
		"Input file name, supported file formats: .vdb")
		->required();
	app.add_option("output,-o,--output", args.outputFilename, 
		"Output file name, supported file formats: .ppm, .exr\n"
		"Must contain a printf expression '%d' in animation mode, but the % replaced by & for compatibility.\n"
		"Alternative specify 'PIPE' to switch to pipe mode, downscaling and animation will be ignored.")
		->required();

	auto downscalePathOpt = app.add_option("--downscale_path", args.downscaleFilename, 
		"Output file for additional downscaled image. If specified, downscale_factor has to be specified as well.\n"
		"Must contain a printf expression '%d' in animation mode, but the % replaced by & for compatibility.");
	auto downscaleFactorOpt = app.add_option("--downscale_factor", args.downscaleFactor, 
		"Downscaling factor, to be used together with --downscale_path")
		->check(CLI::Range(1, 32));
	app.add_option("-d,--depth", args.downscaleDepthFilename, 
		"Output file name for the downscaled normal+depth image, supported file formats: .ppm, .exr.\n"
		"Must contain a printf expression '%d' in animation mode, but the % replaced by & for compatibility.");
	downscalePathOpt->needs(downscaleFactorOpt);
	app.add_option("--highdepth", args.highDepthFilename,
		"Output file name for the high-resolution normal+depth image, supported file formats: .ppm, .exr.\n"
		"Must contain a printf expression '%d' in animation mode, but the % replaced by & for compatibility.");

	//animation
	auto animationOp = app.add_option("-a,--animation", args.animFrames,
		"Enables animation mode: specifies the number of frames to generate. Default: 0 = static image");
	app.add_option("-f,--flow", args.downscaleFlowFilename,
		"Output file name for the downscaled flow image, supported file formats: .ppm, .exr.\n"
		"Must contain a printf expression '%d'. Only used in animation mode, but the % replaced by & for compatibility.\n"
		"Flow is computed from the current frame to the next frame. Hence, the last frame does not contain flow information")
		->needs(animationOp);

	//camera
	app.add_option("--fov", args.cameraFov, 
		"perspective camera field ov view (default: " + std::to_string(args.cameraFov) + ")");
	std::string resolutionStr = "512,512";
	app.add_option("--res,--resolution", resolutionStr, 
		"image resolution (default: " + resolutionStr + ")");
	std::string originStr = "0,0,-2";
	app.add_option("--origin", originStr, 
		"camera origin X,Y,Z (default: " + originStr + ").\n"
		"In animation mode must contain six elements: start X,Y,Z, end X,Y,Z");
	std::string lookatStr = "0,0,0";
	app.add_option("--lookat", lookatStr, 
		"rotate the camera to point to X,Y,Z (default: " + lookatStr + ").\n"
		"In animation mode must contain six elements: start X,Y,Z, end X,Y,Z");
	std::string upStr = "0,1,0";
	app.add_option("--up", upStr, "camera up vector, in combination with --lookat (default: " + upStr + ")");

	//rendering mode
	app.add_set_ignore_case("-m,--mode", args.renderMode, { Args::RenderModeVolume, Args::RenderModeIso, Args::RenderModeConvert })
		->description("Render mode")->required();
	app.add_option("--noshading", args.noShading, 
		"Set to one for no phong shading, color output contains only the albedo.");

	//isosurface
	app.add_option("--isovalue", args.isovalue, 
		"Isovalue in world units for level set ray intersection (default: " + std::to_string(args.isovalue) + ')');
	std::string materialDiffuseStr = "0.7,0.7,0.7";
	app.add_option("--diffuse", materialDiffuseStr, 
		"Iso: material diffuse color R,G,B (default: " + materialDiffuseStr + ")");
	std::string materialSpecularStr = "0.1,0.1,0.1";
	app.add_option("--specular", materialSpecularStr, 
		"Iso: material specular color R,G,B (default: " + materialSpecularStr + ")");
	std::string materialAmbientStr = "0.1,0.1,0.1";
	app.add_option("--ambient", materialAmbientStr, "Iso: material ambient color R,G,B (default: " + materialAmbientStr + ")");
	app.add_option("--exponent", args.materialSpecularExponent, "ISO: specular exponent (default: " + std::to_string(args.materialSpecularExponent) + ")");
	std::string lightStr = "camera";
	app.add_option("--light", lightStr, "Iso: light direction, either X,Y,Z or 'camera' (default: " + lightStr + ")");
	app.add_option("--samples", args.samples, "The number of samples per pixel to reduce alising (default: " + std::to_string(args.samples) + ')')->check(CLI::Range(1, 64));

	//iso effects (unused)
	app.add_option("--ao", args.aoMode, "Ambient Occlusion Mode, supported values: 'screen' and 'world'");
	app.add_option("--aoradius", args.aoRadius, "radius of the ambient occlusion in world space");
	app.add_option("--aosamples", args.aoSamples, "Number of samples for the ambient occlusion");
	app.add_option("--higheffects", args.highEffectsFilename,
		"Output file name for the high-resolution effects image (ambient occlusion + shadow), supported file formats: .ppm, .exr.\n"
		"Must contain a printf expression '%d' in animation mode, but the % replaced by & for compatibility.");

	//pipe
	app.add_option("--pipeOutputFormat", args.pipeOutputFormat,
		"Output format for the pipe mode. Currently the only supported mode is " + args.pipeOutputFormat +
		", which is also the default setting");

	try
	{
		(app).parse((argc), (argv));
	} catch(const CLI::ParseError &e)
	{
		int code = (app).exit(e);
		exit(code);
	}

	//convert parameters
	if (args.animFrames > 1) {
		std::replace(args.outputFilename.begin(), args.outputFilename.end(), '&', '%');
		if (!hasIntFormat(args.outputFilename))
		{
			std::cout << "Animation mode! Output filename must be specified with '%d' or similar." << std::endl;
			exit(-1);
		}
		std::replace(args.highDepthFilename.begin(), args.highDepthFilename.end(), '&', '%');
		if (!args.highDepthFilename.empty() && !hasIntFormat(args.highDepthFilename))
		{
			std::cout << "Animation mode! Highres depth output filename must be specified with '%d' or similar." << std::endl;
			exit(-1);
		}
		std::replace(args.downscaleFilename.begin(), args.downscaleFilename.end(), '&', '%');
		if (!args.downscaleFilename.empty() && !hasIntFormat(args.downscaleFilename))
		{
			std::cout << "Animation mode! Downscale output filename must be specified with '%d' or similar." << std::endl;
			exit(-1);
		}
		std::replace(args.downscaleDepthFilename.begin(), args.downscaleDepthFilename.end(), '&', '%');
		if (!args.downscaleDepthFilename.empty() && !hasIntFormat(args.downscaleDepthFilename))
		{
			std::cout << "Animation mode! Downscale depth output filename must be specified with '%d' or similar." << std::endl;
			exit(-1);
		}
		std::replace(args.downscaleFlowFilename.begin(), args.downscaleFlowFilename.end(), '&', '%');
		if (!args.downscaleFlowFilename.empty() && !hasIntFormat(args.downscaleFlowFilename))
		{
			std::cout << "Animation mode! Downscale flow output filename must be specified with '%d' or similar." << std::endl;
			exit(-1);
		}
	}
	try
	{
		parseStr(resolutionStr, args.resolutionX, args.resolutionY);
	}
	catch (const std::exception& ex) {
		std::cout << "Wrong format for the image resolution, expected w,h: " << ex.what() << std::endl;
		exit(-1);
	}
	try
	{
		if (args.animFrames > 1)
			parseStr(originStr, 
				args.cameraOrigin.x(), args.cameraOrigin.y(), args.cameraOrigin.z(),
				args.cameraOriginEnd.x(), args.cameraOriginEnd.y(), args.cameraOriginEnd.z());
		else
			parseStr(originStr, args.cameraOrigin.x(), args.cameraOrigin.y(), args.cameraOrigin.z());
	}
	catch (const std::exception& ex) {
		std::cout << "Wrong format for the camera origin, expected X,Y,Z: " << ex.what() << std::endl;
		exit(-1);
	}
	try
	{
		if (args.animFrames > 1)
			parseStr(lookatStr, 
				args.cameraLookAt.x(), args.cameraLookAt.y(), args.cameraLookAt.z(),
				args.cameraLookAtEnd.x(), args.cameraLookAtEnd.y(), args.cameraLookAtEnd.z());
		else
			parseStr(lookatStr, args.cameraLookAt.x(), args.cameraLookAt.y(), args.cameraLookAt.z());
	}
	catch (const std::exception& ex) {
		std::cout << "Wrong format for the look-at position, expected X,Y,Z: " << ex.what() << std::endl;
		exit(-1);
	}
	try
	{
		parseStr(upStr, args.cameraUp.x(), args.cameraUp.y(), args.cameraUp.z());
	}
	catch (const std::exception& ex) {
		std::cout << "Wrong format for the up vector, expected X,Y,Z: " << ex.what() << std::endl;
		exit(-1);
	}
	args.cameraUp.normalize();

	try
	{
		parseStr(materialDiffuseStr, args.materialDiffuse.x(), args.materialDiffuse.y(), args.materialDiffuse.z());
	}
	catch (const std::exception& ex) {
		std::cout << "Wrong format for the diffuse color, expected R,G,B: " << ex.what() << std::endl;
		exit(-1);
	}
	try
	{
		parseStr(materialAmbientStr, args.materialAmbient.x(), args.materialAmbient.y(), args.materialAmbient.z());
	}
	catch (const std::exception& ex) {
		std::cout << "Wrong format for the ambient color, expected R,G,B: " << ex.what() << std::endl;
		exit(-1);
	}
	try
	{
		parseStr(materialSpecularStr, args.materialSpecular.x(), args.materialSpecular.y(), args.materialSpecular.z());
	}
	catch (const std::exception& ex) {
		std::cout << "Wrong format for the diffuse color, expected R,G,B: " << ex.what() << std::endl;
		exit(-1);
	}
	if (lightStr == "camera")
		args.cameraLight = true;
	else
	{
		args.cameraLight = false;
		try
		{
			parseStr(lightStr, args.lightDirection.x(), args.lightDirection.y(), args.lightDirection.z());
			args.lightDirection.normalize();
		}
		catch (const std::exception& ex) {
			std::cout << "Wrong format for the light direction, expected X,Y,Z or 'camera': " << ex.what() << std::endl;
			exit(-1);
		}
	}

	//TEST
	std::cout << "input filename: " << args.inputFilename << std::endl;
	std::cout << "output filename: " << args.outputFilename << std::endl;
	std::cout << "fov: " << args.cameraFov << " -> focal length: " << openvdb::tools::PerspectiveCamera::fieldOfViewToFocalLength(args.cameraFov, 41.2) << std::endl;
	std::cout << "resolution: " << args.resolutionX << "x" << args.resolutionY << std::endl;
	std::cout << "camera origin: " << args.cameraOrigin << std::endl;
	std::cout << "camera look-at: " << args.cameraLookAt << std::endl;
	std::cout << "camera up: " << args.cameraUp << std::endl;
	std::cout << "render mode: " << args.renderMode << std::endl;
	std::cout << "isovalue: " << args.isovalue << std::endl;
	std::cout << "material ambient: " << args.materialAmbient << std::endl;
	std::cout << "material diffuse: " << args.materialDiffuse << std::endl;
	std::cout << "material specular: " << args.materialSpecular << std::endl;
	std::cout << "material specular exponent: " << args.materialSpecularExponent << std::endl;
	if (args.cameraLight) std::cout << "light direction: camera origin to camera look-at" << std::endl;
	else std::cout << "light direction: " << args.lightDirection << std::endl;
	std::cout << std::endl;

	return args;
}

//Convert from .vdb to .vdx
void convert(const Args& args)
{
	std::cout << "Convert from " << args.inputFilename << " to " << args.outputFilename << std::endl;
	if (!boost::ends_with(args.inputFilename, ".dat"))
	{
		std::cout << "Error: Input must end in ..dat" << std::endl;
		return;
	}
	if (!boost::ends_with(args.outputFilename, ".vdb"))
	{
		std::cout << "Error: Output must end in .vdb" << std::endl;
		return;
	}

	std::cout << "Load " << args.inputFilename << std::endl;
	openvdb::FloatGrid::Ptr grid = ExternalImporter::importRAW(args.inputFilename, args.downscaleFactor);
	if (grid == nullptr)
		return;

	//grid statistics
	grid->print(std::cout);
	std::cout << "Background value: " << grid->background() << std::endl;
	float minValue, maxValue; grid->evalMinMax(minValue, maxValue);
	std::cout << "Grid min value: " << minValue << ", max value: " << maxValue << std::endl;
	auto gridDim = grid->evalActiveVoxelDim(); auto gridCount = grid->activeVoxelCount();
	std::cout << "Grid number of voxels: " << gridCount << " (" << (gridCount * 100 / (gridDim.x()*gridDim.y()*gridDim.z())) << "%)" << std::endl;
	std::cout << "Grid uniform voxels: " << grid->hasUniformVoxels() << std::endl << std::endl;

	//scale grid to [-0.5,-0.5,-0.5] x [0.5,0.5,0.5]
	auto bbox = grid->evalActiveVoxelBoundingBox();
	auto center = bbox.getCenter();
	std::cout << "Original bounding box (index space): " << bbox.getStart() << " x " << bbox.getEnd() << std::endl;
	auto bboxWorld = grid->transformPtr()->indexToWorld(bbox);
	std::cout << "Original bounding box (world space): " << bboxWorld.min() << " x " << bboxWorld.max() << std::endl;
	auto size = bboxWorld.extents();
	double scale = 1.0 / std::max({ size.x(), size.y(), size.z() });
	grid->transformPtr()->postTranslate(-bboxWorld.getCenter());
	grid->transformPtr()->postScale(scale);
	bboxWorld = grid->transformPtr()->indexToWorld(bbox);
	std::cout << "Transformed bounding box: " << bboxWorld.min() << " x " << bboxWorld.max() << std::endl;
	std::cout << "Grid uniform voxels: " << grid->hasUniformVoxels() << std::endl;
	std::cout << "Transformation: " << grid->constTransform();

	//save grid
	std::cout << "Save grid" << std::endl;
	std::ofstream ofile(args.outputFilename, std::ofstream::binary);
	openvdb::GridPtrVecPtr grids(new openvdb::GridPtrVec);
	grids->push_back(grid);
	openvdb::io::Stream(ofile).write(*grids);
	std::cout << "Done" << std::endl;
}


openvdb::FloatGrid::Ptr loadGrid(const Args& args)
{
	std::cout << "Load Grid" << std::endl;

	openvdb::FloatGrid::Ptr grid;
	openvdb::io::File file(args.inputFilename);
	//No grid was specified by name, retrieve the first float grid from the file.
	file.open(/*delayLoad=*/false);
	openvdb::io::File::NameIterator it = file.beginName();
	openvdb::GridPtrVecPtr grids = file.readAllGridMetadata();
	for (size_t i = 0; i < grids->size(); ++i, ++it) {
		grid = openvdb::gridPtrCast<openvdb::FloatGrid>(grids->at(i));
		if (grid) {
			std::string gridName = *it;
			file.close();
			file.open();
			grid = openvdb::gridPtrCast<openvdb::FloatGrid>(file.readGrid(gridName));
			break;
		}
	}
	if (!grid)
	{
		std::cout << "No scalar, floating-point volume in file " << args.inputFilename << std::endl;
		exit(-2);
	}

	//TEST
	//makeSphere(*grid);

	//grid statistics
	grid->print(std::cout);
	std::cout << "Background value: " << grid->background() << std::endl;
	float minValue, maxValue; grid->evalMinMax(minValue, maxValue);
	std::cout << "Grid min value: " << minValue << ", max value: " << maxValue << std::endl;
	auto gridDim = grid->evalActiveVoxelDim(); auto gridCount = grid->activeVoxelCount();
	std::cout << "Grid number of voxels: " << gridCount << " (" << (gridCount * 100 / (gridDim.x()*gridDim.y()*gridDim.z())) << "%)" << std::endl;
	std::cout << "Grid uniform voxels: " << grid->hasUniformVoxels() << std::endl << std::endl;

	//scale grid to [-0.5,-0.5,-0.5] x [0.5,0.5,0.5]
	auto bbox = grid->evalActiveVoxelBoundingBox();
	auto center = bbox.getCenter();
	std::cout << "Original bounding box (index space): " << bbox.getStart() << " x " << bbox.getEnd() << std::endl;
	auto bboxWorld = grid->transformPtr()->indexToWorld(bbox);
	std::cout << "Original bounding box (world space): " << bboxWorld.min() << " x " << bboxWorld.max() << std::endl;
	auto size = bboxWorld.extents();
	double scale = 1.0 / std::max({ size.x(), size.y(), size.z() });
	grid->transformPtr()->postTranslate(-bboxWorld.getCenter());
	grid->transformPtr()->postScale(scale);
	bboxWorld = grid->transformPtr()->indexToWorld(bbox);
	std::cout << "Transformed bounding box: " << bboxWorld.min() << " x " << bboxWorld.max() << std::endl;
	std::cout << "Grid uniform voxels: " << grid->hasUniformVoxels() << std::endl;
	std::cout << "Transformation: " << grid->constTransform();

	return grid;
}

typedef std::shared_ptr<openvdb::tools::Film> FilmPtr;
std::array<FilmPtr, 3> //RGBA, NxNyNzD, VxVy
render(openvdb::FloatGrid::Ptr grid, const Args& args, 
	const openvdb::Vec3d& currentOrigin, const openvdb::Vec3d& currentLookAt,
	const openvdb::Vec3d& nextOrigin, const const openvdb::Vec3d& nextLookAt,
	float* secondsOut = nullptr)
{
	//std::cout << "Render Image" << std::endl;

	//create output image
	FilmPtr filmRGB = std::make_shared<openvdb::tools::Film>(args.resolutionX, args.resolutionY);
	filmRGB->fill(openvdb::tools::Film::RGBA(0.0,0.0,0.0,0.0));
	FilmPtr filmDnormal = std::make_shared<openvdb::tools::Film>(args.resolutionX, args.resolutionY);
	filmDnormal->fill(openvdb::tools::Film::RGBA(0.0, 0.0, 0.0, 0.0));
	FilmPtr filmVelocities = std::make_shared<openvdb::tools::Film>(args.resolutionX, args.resolutionY);
	filmVelocities->fill(openvdb::tools::Film::RGBA(0.0, 0.0, 0.0, 0.0));

	//create camera
	double aperture = 0.01;
	double focalLength = openvdb::tools::PerspectiveCamera::fieldOfViewToFocalLength(args.cameraFov, aperture);
	openvdb::tools::PerspectiveCamera camera(*filmRGB, 
		openvdb::Vec3R(0), currentOrigin,
		focalLength, aperture);
	camera.lookAt(currentLookAt, args.cameraUp);
	//std::cout << "Camera ray dir top-left: " << camera.getRay(0, 0).dir() << std::endl;
	//std::cout << "Camera ray dir bottom-right: " << camera.getRay(args.resolutionX-1, args.resolutionY-1).dir() << std::endl;

	openvdb::tools::PerspectiveCamera nextCamera(*filmRGB,
		openvdb::Vec3R(0), nextOrigin,
		focalLength, aperture);
	nextCamera.lookAt(nextLookAt, args.cameraUp);

	if (args.renderMode == Args::RenderModeIso)
	{
		//isosurface tracer
		float minValue, maxValue;
		grid->evalMinMax(minValue, maxValue);
		double isovalue = args.isovalue * maxValue;
		openvdb::Vec3R lightDir = args.cameraLight
			? (args.cameraLookAt - args.cameraOrigin)
			: args.lightDirection;
		lightDir.normalize();
		PhongShader<openvdb::tools::BoxSampler> phongShader(
			lightDir, args.materialAmbient, args.materialDiffuse, args.materialSpecular, args.materialSpecularExponent);
		openvdb::tools::MatteShader<openvdb::tools::Film::RGBA, openvdb::tools::BoxSampler> matteShader(
			openvdb::tools::Film::RGBA(args.materialDiffuse.x(), args.materialDiffuse.y(), args.materialDiffuse.z()));
		openvdb::tools::BaseShader* shader = args.noShading 
		? static_cast<openvdb::tools::BaseShader*>(&matteShader) 
		: static_cast<openvdb::tools::BaseShader*>(&phongShader);
		openvdb::tools::IsoVolumeRayIntersector<openvdb::FloatGrid> intersector(
			*grid, static_cast<typename openvdb::FloatGrid::ValueType>(isovalue));
		openvdb::tools::IsoVolumeRayTracer<openvdb::FloatGrid, openvdb::tools::IsoVolumeRayIntersector<openvdb::FloatGrid>> tracer(
			intersector, *shader, camera, args.samples, 42);
		tracer.setDepthNormalFilm(filmDnormal.get());
		tracer.setFlowFilm(filmVelocities.get(), &nextCamera);

		auto start = std::chrono::high_resolution_clock::now();
		tracer.render(true);
		auto finish = std::chrono::high_resolution_clock::now();
		if (secondsOut) *secondsOut = std::chrono::duration<double>(finish - start).count();
	}
	else
	{
		//volume tracer
		//TODO

		using IntersectorType = openvdb::tools::VolumeRayIntersector<openvdb::FloatGrid>;
		IntersectorType intersector(*grid);

		openvdb::tools::VolumeRender<IntersectorType> renderer(intersector, camera);
		openvdb::Vec3R lightDir = args.cameraLight
			? (args.cameraLookAt - args.cameraOrigin)
			: args.lightDirection;
		lightDir.normalize();
		renderer.setLightDir(lightDir.x(), lightDir.y(), lightDir.z());
		renderer.setLightColor(1, 1, 1);
		renderer.setPrimaryStep(1);
		renderer.setShadowStep(2);
		renderer.setScattering(1.5, 1.5, 1.5);
		renderer.setAbsorption(0.1, 0.1, 0.1);
		renderer.setLightGain(0.2);
		renderer.setCutOff(0.005);

		auto start = std::chrono::high_resolution_clock::now();
		renderer.render(true);
		auto finish = std::chrono::high_resolution_clock::now();
		if (secondsOut) *secondsOut = std::chrono::duration<double>(finish - start).count();
	}

	int numOn = 0, numOff = 0;
	for (int i=0; i< args.resolutionX; ++i)
		for (int j=0; j< args.resolutionY; ++j)
		{
			if (filmRGB->pixel(i, j).a == 0)
				numOff++;
			else
				numOn++;
		}
	//std::cout << "Num on: " << numOn << ", numOff: " << numOff << std::endl;

	return { filmRGB, filmDnormal, filmVelocities };
}

void saveImage(const openvdb::tools::Film& image, const std::string& outputFilename, int numChannels)
{
	std::cout << "Save Image to " << outputFilename << std::endl;
	if (boost::iends_with(outputFilename, ".ppm")) {
		// Save as PPM (fast, but large file size).
		std::string filename = outputFilename;
		filename.erase(filename.size() - 4); // strip .ppm extension
		const_cast<openvdb::tools::Film*>(&image)->savePPM(filename); //savePPM is non-const!!
	}
	else if (boost::iends_with(outputFilename, ".exr")) {
		// Save as EXR (slow, but small file size).
		Imf::setGlobalThreadCount(8);
		Imf::Header header(int(image.width()), int(image.height()));
		header.compression() = Imf::ZIP_COMPRESSION;
		header.channels().insert("R", Imf::Channel(Imf::FLOAT));
		if (numChannels >= 2) header.channels().insert("G", Imf::Channel(Imf::FLOAT));
		if (numChannels >= 3) header.channels().insert("B", Imf::Channel(Imf::FLOAT));
		if (numChannels >= 4) header.channels().insert("A", Imf::Channel(Imf::FLOAT));
		const size_t pixelBytes = sizeof(openvdb::tools::Film::RGBA), rowBytes = pixelBytes * image.width();
		openvdb::tools::Film::RGBA& pixel0 = const_cast<openvdb::tools::Film::RGBA*>(image.pixels())[0];
		Imf::FrameBuffer framebuffer;
		framebuffer.insert("R",
			Imf::Slice(Imf::FLOAT, reinterpret_cast<char*>(&pixel0.r), pixelBytes, rowBytes));
		if (numChannels >= 2)
			framebuffer.insert("G",
				Imf::Slice(Imf::FLOAT, reinterpret_cast<char*>(&pixel0.g), pixelBytes, rowBytes));
		if (numChannels >= 3)
			framebuffer.insert("B",
				Imf::Slice(Imf::FLOAT, reinterpret_cast<char*>(&pixel0.b), pixelBytes, rowBytes));
		if (numChannels >=4)
			framebuffer.insert("A",
				Imf::Slice(Imf::FLOAT, reinterpret_cast<char*>(&pixel0.a), pixelBytes, rowBytes));

		Imf::OutputFile imgFile(outputFilename.c_str(), header);
		imgFile.setFrameBuffer(framebuffer);
		imgFile.writePixels(int(image.height()));
	}
	else {
		std::cout << "unsupported image file format (" << outputFilename << ")" << std::endl;
	}
}

void renderSingle(openvdb::FloatGrid::Ptr grid, const Args& args)
{
	//high-res
	FilmPtr image = render(grid, args, 
		args.cameraOrigin, args.cameraLookAt,
		args.cameraOrigin, args.cameraLookAt)[0];
	saveImage(*image, args.outputFilename, 4);

	//low-res
	if (!args.downscaleFilename.empty())
	{
		Args args2 = args;
		args2.samples = 1;
		args2.resolutionX = args.resolutionX / args.downscaleFactor;
		args2.resolutionY = args.resolutionY / args.downscaleFactor;
		auto images = render(grid, args2, 
			args.cameraOrigin, args.cameraLookAt, 
			args.cameraOrigin, args.cameraLookAt);
		saveImage(*images[0], args.downscaleFilename, 4);
		if (!args.downscaleDepthFilename.empty())
			saveImage(*images[1], args.downscaleDepthFilename, 4);
	}
}

openvdb::Vec3d interpolate(const openvdb::Vec3d& a, const openvdb::Vec3d& b, double alpha)
{
	return a * (1 - alpha) + b * alpha;
}
void renderAnimation(openvdb::FloatGrid::Ptr grid, const Args& args)
{
	for (int frame=0; frame < args.animFrames; ++frame)
	{
		openvdb::Vec3d currentCameraOrigin = interpolate(
			args.cameraOrigin, args.cameraOriginEnd,
			frame / double(args.animFrames - 1));
		openvdb::Vec3d nextCameraOrigin = interpolate(
			args.cameraOrigin, args.cameraOriginEnd,
			(frame+1) / double(args.animFrames - 1));
		openvdb::Vec3d currentCameraLookAt = interpolate(
			args.cameraLookAt, args.cameraLookAtEnd,
			frame / double(args.animFrames - 1));
		openvdb::Vec3d nextCameraLookAt = interpolate(
			args.cameraLookAt, args.cameraLookAtEnd,
			(frame + 1) / double(args.animFrames - 1));

		//high-res
		float time;
		auto images = render(grid, args,
			currentCameraOrigin, currentCameraLookAt,
			nextCameraOrigin, nextCameraLookAt,
			&time);
		saveImage(*images[0], tfm::format(args.outputFilename, frame), 4);
		if (!args.highDepthFilename.empty())
			saveImage(*images[1], tfm::format(args.highDepthFilename, frame), 4);
		std::cout << "Time to render high resolution: " << time << " sec" << std::endl;

		//low-res
		if (!args.downscaleFilename.empty())
		{
			Args args2 = args;
			args2.samples = 1;
			args2.resolutionX = args.resolutionX / args.downscaleFactor;
			args2.resolutionY = args.resolutionY / args.downscaleFactor;
			auto images = render(grid, args2,
				currentCameraOrigin, currentCameraLookAt,
				nextCameraOrigin, nextCameraLookAt,
				&time);
			saveImage(*images[0], tfm::format(args.downscaleFilename, frame), 4);
			if (!args.downscaleDepthFilename.empty())
				saveImage(*images[1], tfm::format(args.downscaleDepthFilename, frame), 4);
			if (!args.downscaleFlowFilename.empty())
				saveImage(*images[2], tfm::format(args.downscaleFlowFilename, frame), 4);
			std::cout << "Time to render low resolution: " << time << " sec" << std::endl;
		}
	}
}

void renderPipe(openvdb::FloatGrid::Ptr grid, Args args)
{
	std::cout << "Enter Pipe mode and wait for commands" << std::endl;

	openvdb::Vec3d lastOrigin = args.cameraOrigin;
	openvdb::Vec3d lastLookAt = args.cameraLookAt;

	if (args.pipeOutputFormat != "RGBMXYZDUV")
	{
		std::cout << "Unsupported pipe output format '" << args.pipeOutputFormat
			<< "', only 'RGBMXYZDUV' is supported" << std::endl;
		return;
	}
#define IDX(c, y, x) ((x) + args.resolutionX * ((y) + args.resolutionY * (c)))

	static const int32_t scale = 1000;

	while (true)
	{
		std::string command;
		std::getline(std::cin, command);
		if (command=="exit")
		{
			std::cout << "Exit program" << std::endl;
			return;
		} else if (command=="render")
		{
			//render
			float time;
			auto images = render(grid, args,
				args.cameraOrigin, args.cameraLookAt,
				lastOrigin, lastLookAt, &time); //use last camera as next camera and invert flow
			//write to stdout (cerr)
			std::vector<float> output(args.resolutionX * args.resolutionY * 12 + 1, 42.0f);
//#pragma omp parallel for
			for (int y=0; y<args.resolutionY; ++y) 
				for (int x=0; x<args.resolutionX; ++x)
			{
				output[IDX(0, y, x)] = static_cast<float>(images[0]->pixel(x, y).r); //color
				output[IDX(1, y, x)] = static_cast<float>(images[0]->pixel(x, y).g);
				output[IDX(2, y, x)] = static_cast<float>(images[0]->pixel(x, y).b);
				output[IDX(3, y, x)] = static_cast<float>(images[0]->pixel(x, y).a); //mask
				output[IDX(4, y, x)] = static_cast<float>(images[1]->pixel(x, y).r); //normal
				output[IDX(5, y, x)] = static_cast<float>(images[1]->pixel(x, y).g);
				output[IDX(6, y, x)] = static_cast<float>(images[1]->pixel(x, y).b);
				output[IDX(7, y, x)] = static_cast<float>(images[1]->pixel(x, y).a); //depth
				output[IDX(8, y, x)] = -static_cast<float>(images[2]->pixel(x, y).r); //flow
				output[IDX(9, y, x)] = -static_cast<float>(images[2]->pixel(x, y).g);
				output[IDX(10,y, x)] = 1; //Ambient occlusion, not supported
				output[IDX(11,y, x)] = 0; //shadow, not supported
			}
			output[output.size() - 1] = time;
			//for (int c = 0; c < 10; ++c) for (int y = 0; y < args.resolutionY; ++y) {
			//	std::cerr.write(reinterpret_cast<const char*>(output.data() + args.resolutionX * (y + args.resolutionY*c)),
			//		args.resolutionX * sizeof(int32_t));
			//	std::cerr.flush();
			//}
			std::cerr.write(reinterpret_cast<const char*>(output.data()), output.size() * sizeof(float));
			std::cerr.flush();
			//save old camera position for flow
			lastOrigin = args.cameraOrigin;
			lastLookAt = args.cameraLookAt;
		} else
		{ //command=value
			auto idx = command.find('=');
			if (idx == std::string::npos)
			{
				std::cout << "Unknown command format: " << command << ", exit" << std::endl;
				return;
			}
			std::string cmd = command.substr(0, idx);
			std::string value = command.substr(idx + 1);
			if (cmd == "cameraOrigin")
				parseStr(value, args.cameraOrigin.x(), args.cameraOrigin.y(), args.cameraOrigin.z());
			else if (cmd == "cameraLookAt")
				parseStr(value, args.cameraLookAt.x(), args.cameraLookAt.y(), args.cameraLookAt.z());
			else if (cmd == "cameraUp")
				parseStr(value, args.cameraUp.x(), args.cameraUp.y(), args.cameraUp.z());
			else if (cmd == "cameraFoV")
				parseStr(value, args.cameraFov);
			else if (cmd == "resolution")
				parseStr(value, args.resolutionX, args.resolutionY);
			else if (cmd == "isovalue")
				parseStr(value, args.isovalue);
			else if (cmd == "unshaded")
				parseStr(value, args.noShading);
			else if (cmd == "aosamples")
				(void)0;
			else if (cmd == "aoradius")
				(void)0;
			else if (cmd == "viewport") //minX, minY, maxX, maxY
				(void)0;
			//TODO: more commands
			else {
				std::cout << "Unknown command: '" << cmd << "', exit" << std::endl;
				return;
			}
		}
	}
}

int main(int argc, const char* argv[])
{	
	//https://stackoverflow.com/questions/36987437/data-corruption-piping-between-c-and-python
	_setmode(_fileno(stderr), O_BINARY);

	Args args = parseArguments(argc, argv);
	openvdb::initialize();

	if (args.renderMode == Args::RenderModeConvert) {
		convert(args);
		return 0;
	}

	openvdb::FloatGrid::Ptr grid = loadGrid(args);

	if (args.outputFilename == "PIPE")
		renderPipe(grid, args);
	else if (args.animFrames > 1)
		renderAnimation(grid, args);
	else
		renderSingle(grid, args);

}
