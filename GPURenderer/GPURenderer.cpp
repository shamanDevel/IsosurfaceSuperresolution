// GPURenderer.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"

#include <iostream>
#include <iomanip>
#include <sstream>
#include <array>
#include <algorithm>
#include <random>
#include <cmath>
#include <chrono>
#include <CLI11.hpp>
#include <tinyformat.h>
#include <boost/algorithm/string/predicate.hpp>

#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfFrameBuffer.h>
#include <OpenEXR/ImfHeader.h>
#include <OpenEXR/ImfOutputFile.h>
#include <OpenEXR/ImfPixelType.h>

#include <gvdb/gvdb.h>
#include <vector_types.h>
#include <vector_functions.h>

#include <fcntl.h>
#include <io.h>

#include "ShadowFX.h"
#include "Vdb2Vbx.h"

using namespace nvdb;

VolumeGVDB gvdb;
std::unique_ptr<Camera3D> camera;
CUmodule cuCustom;
CUfunction cuIsoKernel;
CUdeviceptr kernelLightDir;
CUdeviceptr kernelAmbientColor;
CUdeviceptr kernelDiffuseColor;
CUdeviceptr kernelSpecularColor;
CUdeviceptr kernelSpecularExponent;
CUdeviceptr kernelCurrentViewMatrix;
CUdeviceptr kernelNextViewMatrix;
CUdeviceptr kernelNormalMatrix;
CUdeviceptr kernelAoSamples;
CUdeviceptr kernelAoHemisphere;
CUdeviceptr kernelAoRandomRotations;
CUdeviceptr kernelAoRadius;
CUdeviceptr kernelViewport;
#define MAX_AMBIENT_OCCLUSION_SAMPLES 512  //same as in render_kernel.cu
#define AMBIENT_OCCLUSION_RANDOM_ROTATIONS 4

namespace nvdb {
	std::ostream& operator<<(std::ostream& o, const Vector3DF& v)
	{
		o << v.x << "," << v.y << "," << v.z;
		return o;
	}
}

template<typename Arg>
void parseStrImpl(const std::string& str, std::size_t pos, Arg& arg)
{
	if (str.find(',', pos) != std::string::npos)
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
	}
	catch (std::exception& ex)
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
	int downscaleFactor = 0;
	std::string downscaleFilename;
	std::string downscaleDepthFilename;
	std::string highDepthFilename;
	//animation
	int animFrames = 0;
	std::string downscaleFlowFilename;
	Vector3DF cameraOriginEnd = { 0,0,-1 };
	Vector3DF cameraLookAtEnd = { 0,0,0 };
	//camera
	int resolutionX = 512;
	int resolutionY = 512;
	double cameraFov = 45;
	Vector3DF cameraOrigin = { 0,0,-1 };
	Vector3DF cameraLookAt = { 0,0,0 };
	Vector3DF cameraUp = { 0,1,0 };
	int4 viewport = make_int4(0, 0, resolutionX, resolutionY);
	//render mode
	static const std::string RenderModeVolume;
	static const std::string RenderModeIso;
	static const std::string RenderModeConvert;
	std::string renderMode;
	int noShading = 0;
	//isosurface
	double isovalue = 0.0;
	Vector3DF materialDiffuse = { 0.7, 0.7, 0.7 };
	Vector3DF materialSpecular = { 1,1,1 };
	Vector3DF materialAmbient = { 0.1,0.1,0.1 };
	int materialSpecularExponent = 32;
	bool cameraLight = true; //true: light along camera view direction
	Vector3DF lightDirection = { 0,0,0 };
	int samples = 1;
	//iso effects
	static const std::string AOScreen;
	static const std::string AOWorld;
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
const std::string Args::AOScreen = "screen";
const std::string Args::AOWorld = "world";

Args parseArguments(int argc, const char* argv[])
{
	CLI::App app{ "CPU Volume Renderer" };
	Args args;

	//input and output
	app.add_option("input,-i,--input", args.inputFilename,
		"Input file name, supported file formats: .vdb")
		->required();// ->check(CLI::ExistingFile);
	app.add_option("output,-o,--output", args.outputFilename,
		"Output file name, supported file formats: .ppm, .exr, .npy\n"
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

	//iso effects
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
	}
	catch (const CLI::ParseError &e)
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
		std::replace(args.highEffectsFilename.begin(), args.highEffectsFilename.end(), '&', '%');
		if (!args.highEffectsFilename.empty() && !hasIntFormat(args.highEffectsFilename))
		{
			std::cout << "Animation mode! Highres effects output filename must be specified with '%d' or similar." << std::endl;
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
		args.viewport = make_int4(0, 0, args.resolutionX, args.resolutionY);
	}
	catch (const std::exception& ex) {
		std::cout << "Wrong format for the image resolution, expected w,h: " << ex.what() << std::endl;
		exit(-1);
	}
	try
	{
		if (args.animFrames > 1)
			parseStr(originStr,
				args.cameraOrigin.x, args.cameraOrigin.y, args.cameraOrigin.z,
				args.cameraOriginEnd.x, args.cameraOriginEnd.y, args.cameraOriginEnd.z);
		else
			parseStr(originStr, args.cameraOrigin.x, args.cameraOrigin.y, args.cameraOrigin.z);
	}
	catch (const std::exception& ex) {
		std::cout << "Wrong format for the camera origin, expected X,Y,Z: " << ex.what() << std::endl;
		exit(-1);
	}
	try
	{
		if (args.animFrames > 1)
			parseStr(lookatStr,
				args.cameraLookAt.x, args.cameraLookAt.y, args.cameraLookAt.z,
				args.cameraLookAtEnd.x, args.cameraLookAtEnd.y, args.cameraLookAtEnd.z);
		else
			parseStr(lookatStr, args.cameraLookAt.x, args.cameraLookAt.y, args.cameraLookAt.z);
	}
	catch (const std::exception& ex) {
		std::cout << "Wrong format for the look-at position, expected X,Y,Z: " << ex.what() << std::endl;
		exit(-1);
	}
	try
	{
		parseStr(upStr, args.cameraUp.x, args.cameraUp.y, args.cameraUp.z);
	}
	catch (const std::exception& ex) {
		std::cout << "Wrong format for the up vector, expected X,Y,Z: " << ex.what() << std::endl;
		exit(-1);
	}
	args.cameraUp.Normalize();

	try
	{
		parseStr(materialDiffuseStr, args.materialDiffuse.x, args.materialDiffuse.y, args.materialDiffuse.z);
	}
	catch (const std::exception& ex) {
		std::cout << "Wrong format for the diffuse color, expected R,G,B: " << ex.what() << std::endl;
		exit(-1);
	}
	try
	{
		parseStr(materialAmbientStr, args.materialAmbient.x, args.materialAmbient.y, args.materialAmbient.z);
	}
	catch (const std::exception& ex) {
		std::cout << "Wrong format for the ambient color, expected R,G,B: " << ex.what() << std::endl;
		exit(-1);
	}
	try
	{
		parseStr(materialSpecularStr, args.materialSpecular.x, args.materialSpecular.y, args.materialSpecular.z);
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
			parseStr(lightStr, args.lightDirection.x, args.lightDirection.y, args.lightDirection.z);
			args.lightDirection.Normalize();
		}
		catch (const std::exception& ex) {
			std::cout << "Wrong format for the light direction, expected X,Y,Z or 'camera': " << ex.what() << std::endl;
			exit(-1);
		}
	}

	//TEST
	std::cout << "input filename: " << args.inputFilename << std::endl;
	std::cout << "output filename: " << args.outputFilename << std::endl;
	std::cout << "fov: " << args.cameraFov << std::endl;
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
	if (!boost::ends_with(args.inputFilename, ".vdb") &&
		!boost::ends_with(args.inputFilename, ".dat"))
	{
		std::cout << "Error: Input must end in .vdb or .dat" << std::endl;
		return;
	}
	if (!boost::ends_with(args.outputFilename, ".vbx"))
	{
		std::cout << "Error: Output must end in .vbx" << std::endl;
		return;
	}

	std::cout << "Start GVDB" << std::endl;
	gvdb.SetVerbose(true);		// enable/disable console output from gvdb
	gvdb.SetCudaDevice(GVDB_DEV_FIRST);
	gvdb.Initialize();

	std::cout << "Load " << args.inputFilename << std::endl;
	gvdb.SetChannelDefault(64, 64, 64);
	if (boost::ends_with(args.inputFilename, ".vdb")) {
		//if (!gvdb.LoadVDB(args.inputFilename)) {                	// Load OpenVDB format			
		if (!Vdb2Vbx::LoadVDB(gvdb, args.inputFilename)) {
			gerror();
		}
	}

	//grid statistics
	Vector3DF objmin, objmax, voxmin, voxmax, voxsize, voxres;
	gvdb.getDimensions(objmin, objmax, voxmin, voxmax, voxsize, voxres);
	std::cout << "objmin=" << objmin << std::endl;
	std::cout << "objmax=" << objmax << std::endl;
	std::cout << "voxmin=" << voxmin << std::endl;
	std::cout << "voxmax=" << voxmax << std::endl;
	std::cout << "voxsize=" << voxsize << std::endl;
	std::cout << "voxres=" << voxres << std::endl;

	std::cout << "Save " << args.outputFilename << std::endl;
	gvdb.SaveVBX(args.outputFilename);
}

void initGVDB()
{
	std::cout << "Start GVDB" << std::endl;
#ifdef NDEBUG
	gvdb.SetDebug(false);
#else
	gvdb.SetDebug(true);
#endif
	gvdb.SetVerbose(true);
	gvdb.SetProfile(false, true);
	gvdb.SetCudaDevice(GVDB_DEV_FIRST);
	gvdb.Initialize();
}

void loadGrid(const Args& args)
{
	if (!boost::ends_with(args.inputFilename, ".vbx"))
	{
		std::cout << "Error: Input must end in .vbx" << std::endl;
		exit(-1);
	}
	std::cout << "Load " << args.inputFilename << std::endl;
	if (!gvdb.LoadVBX(args.inputFilename)) {
		gerror();
	}

	//grid statistics and transformation
	Vector3DF objmin, objmax, voxmin, voxmax, voxsize, voxres;
	gvdb.getDimensions(objmin, objmax, voxmin, voxmax, voxsize, voxres);
	std::cout << "objmin=" << objmin << std::endl;
	std::cout << "objmax=" << objmax << std::endl;
	std::cout << "voxmin=" << voxmin << std::endl;
	std::cout << "voxmax=" << voxmax << std::endl;
	std::cout << "voxsize=" << voxsize << std::endl;
	std::cout << "voxres=" << voxres << std::endl;
	Vector3DF invcenter = (objmax + objmin) * -0.5f;
	float scale = 0.5 / std::max({ objmax.x - objmin.x, objmax.y - objmin.y, objmax.z - objmin.z });
	gvdb.SetTransform(invcenter, Vector3DF(scale, scale, scale), Vector3DF(0, 0, 0), Vector3DF(0, 0, 0));
}

bool cudaCheck(CUresult status, const char* msg)
{
	if (status != CUDA_SUCCESS) {
		const char* stat = "";
		cuGetErrorString(status, &stat);
		std::cout << "CUDA ERROR: " << stat << "(in " << msg << ")" << std::endl;
		exit(-1);
		return false;
	}
	return true;
}

//Computes ambient occlusion parameters
void computeAmbientOcclusionParameters(const Args& args)
{
	static std::default_random_engine rnd;
	static std::uniform_real_distribution<float> distr(0.0f, 1.0f);
	//samples
	int samples = args.aoMode == Args::AOWorld ? args.aoSamples : 0;
	samples = std::max(0, std::min(MAX_AMBIENT_OCCLUSION_SAMPLES - 1, samples));
	cudaCheck(cuMemcpyHtoD(kernelAoSamples, &samples, sizeof(int)), "cuMemcpyHtoD");
	//radius
	cudaCheck(cuMemcpyHtoD(kernelAoRadius, &args.aoRadius, sizeof(float)), "cuMemcpyHtoD");
	//samples on a hemisphere
	std::vector<float4> aoHemisphere(MAX_AMBIENT_OCCLUSION_SAMPLES, make_float4(0, 0, 0, 0));
	for (int i=0; i<MAX_AMBIENT_OCCLUSION_SAMPLES; ++i)
	{
		float u1 = distr(rnd);
		float u2 = distr(rnd);
		float r = std::sqrt(u1);
		float theta = 2 * M_PI * u2;
		float x = r * std::cos(theta);
		float y = r * std::sin(theta);
		float scale = distr(rnd);
		scale = 0.1 + 0.9 * scale * scale;
		aoHemisphere[i] = make_float4(x*scale, y*scale, std::sqrt(1 - u1)*scale, 0);
	}
	cudaCheck(cuMemcpyHtoD(
		kernelAoHemisphere, 
		aoHemisphere.data(), 
		sizeof(float)*4* MAX_AMBIENT_OCCLUSION_SAMPLES),
		"cuMemcpyHtoD");
	//random rotation vectors
	std::vector<float4> aoRandomRotations(AMBIENT_OCCLUSION_RANDOM_ROTATIONS*AMBIENT_OCCLUSION_RANDOM_ROTATIONS);
	for (int i=0; i< AMBIENT_OCCLUSION_RANDOM_ROTATIONS*AMBIENT_OCCLUSION_RANDOM_ROTATIONS; ++i)
	{
		float x = distr(rnd) * 2 - 1;
		float y = distr(rnd) * 2 - 1;
		float linv = 1.0f / sqrt(x*x + y * y);
		aoRandomRotations[i] = make_float4(x*linv, y*linv, 0, 0);
	}
	cudaCheck(cuMemcpyHtoD(
		kernelAoRandomRotations,
		aoRandomRotations.data(),
		sizeof(float) * 4 * AMBIENT_OCCLUSION_RANDOM_ROTATIONS*AMBIENT_OCCLUSION_RANDOM_ROTATIONS),
		"cuMemcpyHtoD");
}

void initRendering(const Args& args)
{
	Scene* scn = gvdb.getScene();
	//camera
	camera = std::make_unique<Camera3D>();
	camera->setFov(args.cameraFov);
	camera->setAspect(float(args.resolutionX) / float(args.resolutionY));
	camera->setPos(args.cameraOrigin.x, args.cameraOrigin.y, args.cameraOrigin.z);
	camera->setToPos(args.cameraLookAt.x, args.cameraLookAt.y, args.cameraLookAt.z);
	scn->SetCamera(camera.get());
	scn->SetRes(args.resolutionX, args.resolutionY);
	//output
	gvdb.AddRenderBuf(0, args.resolutionX, args.resolutionY, 12*sizeof(float));

	//load rendering kernel
	std::cout << "Load rendering kernel" << std::endl;
	cudaCheck(cuModuleLoad(&cuCustom, "render_kernel.ptx"), "cuModuleLoad (render_custom)");
	cudaCheck(cuModuleGetFunction(&cuIsoKernel, cuCustom, "custom_iso_kernel"), "cuModuleGetFunction (raycast_kernel)");
	gvdb.SetModule(cuCustom);

	//grap rendering settings
	cudaCheck(cuModuleGetGlobal(&kernelLightDir, NULL, cuCustom, "lightDir"), "cuModuleGetGlobal");
	cudaCheck(cuModuleGetGlobal(&kernelAmbientColor, NULL, cuCustom, "ambientColor"), "cuModuleGetGlobal");
	cudaCheck(cuModuleGetGlobal(&kernelDiffuseColor, NULL, cuCustom, "diffuseColor"), "cuModuleGetGlobal");
	cudaCheck(cuModuleGetGlobal(&kernelSpecularColor, NULL, cuCustom, "specularColor"), "cuModuleGetGlobal");
	cudaCheck(cuModuleGetGlobal(&kernelSpecularExponent, NULL, cuCustom, "specularExponent"), "cuModuleGetGlobal");
	cudaCheck(cuModuleGetGlobal(&kernelCurrentViewMatrix, NULL, cuCustom, "currentViewMatrix"), "cuModuleGetGlobal");
	cudaCheck(cuModuleGetGlobal(&kernelNextViewMatrix, NULL, cuCustom, "nextViewMatrix"), "cuModuleGetGlobal");
	cudaCheck(cuModuleGetGlobal(&kernelNormalMatrix, NULL, cuCustom, "normalMatrix"), "cuModuleGetGlobal");
	cudaCheck(cuModuleGetGlobal(&kernelAoSamples, NULL, cuCustom, "aoSamples"), "cuModuleGetGlobal");
	cudaCheck(cuModuleGetGlobal(&kernelAoHemisphere, NULL, cuCustom, "aoHemisphere"), "cuModuleGetGlobal");
	cudaCheck(cuModuleGetGlobal(&kernelAoRandomRotations, NULL, cuCustom, "aoRandomRotations"), "cuModuleGetGlobal");
	cudaCheck(cuModuleGetGlobal(&kernelAoRadius, NULL, cuCustom, "aoRadius"), "cuModuleGetGlobal");
	cudaCheck(cuModuleGetGlobal(&kernelViewport, NULL, cuCustom, "viewport"), "cuModuleGetGlobal");
	computeAmbientOcclusionParameters(args);
}

std::string getPointerType(const void* ptr)
{
	cudaPointerAttributes attr = {};
	cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
	if (err != cudaSuccess)
		return "ERROR!";
	if (attr.type == cudaMemoryTypeDevice)
		return "device";
	else if (attr.type == cudaMemoryTypeHost)
		return "host";
	else if (attr.type == cudaMemoryTypeManaged)
		return "managed";
	else if (attr.type == cudaMemoryTypeUnregistered)
		return "unregistered";
	else
		return "unknown";
}
std::string getPointerType(CUdeviceptr ptr)
{
	return getPointerType(reinterpret_cast<void*>(ptr));
}

std::vector<float> render(const Args& args,
	const Vector3DF& currentOrigin, const Vector3DF& currentLookAt,
	const Vector3DF& nextOrigin, const const Vector3DF& nextLookAt,
	float* secondsOut = nullptr,
	CUdeviceptr targetDevice = 0) //if !=0, output is copied into this memory instead of returns as vector
{
	Scene* scn = gvdb.getScene();
	bool resize = args.resolutionX != scn->getRes().x || args.resolutionY != scn->getRes().y;
	//camera
	camera->setFov(args.cameraFov);
	camera->setAspect(float(args.resolutionX) / float(args.resolutionY));
	camera->up_dir = args.cameraUp;
	camera->setPos(currentOrigin.x, currentOrigin.y, currentOrigin.z);
	camera->setToPos(currentLookAt.x, currentLookAt.y, currentLookAt.z);
	scn->SetRes(args.resolutionX, args.resolutionY);
	Matrix4F modelViewProj = camera->getFullProjMatrix(); 
	modelViewProj *= camera->getViewMatrix();
	modelViewProj.Transpose();
	cudaCheck(cuMemcpyHtoD(kernelCurrentViewMatrix, modelViewProj.data, sizeof(float) * 16), "cuMemcpyHtoD");
	Camera3D nextCamera = *camera; nextCamera.Copy(*camera);
	nextCamera.setPos(nextOrigin.x, nextOrigin.y, nextOrigin.z);
	nextCamera.setToPos(nextLookAt.x, nextLookAt.y, nextLookAt.z);
	modelViewProj = nextCamera.getFullProjMatrix(); 
	modelViewProj *= nextCamera.getViewMatrix();
	modelViewProj.Transpose();
	cudaCheck(cuMemcpyHtoD(kernelNextViewMatrix, modelViewProj.data, sizeof(float) * 16), "cuMemcpyHtoD");
	Matrix4F normalMatrix = camera->getViewMatrix();
	normalMatrix.Transpose();
	cudaCheck(cuMemcpyHtoD(kernelNormalMatrix, normalMatrix.data, sizeof(float) * 16), "cuMemcpyHtoD");
	cudaCheck(cuMemcpyHtoD(kernelViewport, &args.viewport, sizeof(int) * 4), "cuMemcpyHtoD");
	//light and color
	Vector3DF lightDir = args.cameraLight
		? (Vector3DF(args.cameraLookAt) - args.cameraOrigin)
		: args.lightDirection;
	lightDir.Normalize();
	cudaCheck(cuMemcpyHtoD(kernelLightDir, &lightDir.x, sizeof(float) * 3), "cuMemcpyHtoD");
	cudaCheck(cuMemcpyHtoD(kernelAmbientColor, &args.materialAmbient.x, sizeof(float) * 3), "cuMemcpyHtoD");
	cudaCheck(cuMemcpyHtoD(kernelDiffuseColor, &args.materialDiffuse.x, sizeof(float) * 3), "cuMemcpyHtoD");
	cudaCheck(cuMemcpyHtoD(kernelSpecularColor, &args.materialSpecular.x, sizeof(float) * 3), "cuMemcpyHtoD");
	cudaCheck(cuMemcpyHtoD(kernelSpecularExponent, &args.materialSpecularExponent, sizeof(int)), "cuMemcpyHtoD");
	//ambient occlusion
	//samples
	int aoSamples = args.aoMode == Args::AOWorld ? args.aoSamples : 0;
	aoSamples = std::max(0, std::min(MAX_AMBIENT_OCCLUSION_SAMPLES, aoSamples));
	cudaCheck(cuMemcpyHtoD(kernelAoSamples, &aoSamples, sizeof(int)), "cuMemcpyHtoD");
	cudaCheck(cuMemcpyHtoD(kernelAoRadius, &args.aoRadius, sizeof(float)), "cuMemcpyHtoD");

	//output
	if (resize)
		gvdb.ResizeRenderBuf(0, args.resolutionX, args.resolutionY, 12 * sizeof(float));
	
	if (args.renderMode == Args::RenderModeIso) {
		//iso and step
		scn->SetVolumeRange(args.isovalue, 0.0f, 1.0f);
		scn->SetSteps(0.05, 16, 0.05);

		//render
		cudaDeviceSynchronize();
		auto start = std::chrono::high_resolution_clock::now();
		gvdb.RenderKernel(cuIsoKernel, 0, 0);
		cudaDeviceSynchronize();
		auto finish = std::chrono::high_resolution_clock::now();
		if (secondsOut) *secondsOut = std::chrono::duration<double>(finish-start).count();
	} else if (args.renderMode == Args::RenderModeVolume)
	{
		scn->SetSteps(.25, 16, .25);				// Set raycasting steps
		scn->SetExtinct(-1.0f, 1.0f, 0.0f);		// Set volume extinction
		scn->SetVolumeRange(0.1f, 0.0f, .5f);	// Set volume value range
		scn->SetCutoff(0.005f, 0.005f, 0.0f);
		scn->SetBackgroundClr(0.1f, 0.2f, 0.4f, 1.0);
		scn->LinearTransferFunc(0.00f, 0.25f, Vector4DF(0, 0, 0, 0), Vector4DF(1, 0, 0, 0.05f));
		scn->LinearTransferFunc(0.25f, 0.50f, Vector4DF(1, 0, 0, 0.05f), Vector4DF(1, .5f, 0, 0.1f));
		scn->LinearTransferFunc(0.50f, 0.75f, Vector4DF(1, .5f, 0, 0.1f), Vector4DF(1, 1, 0, 0.15f));
		scn->LinearTransferFunc(0.75f, 1.00f, Vector4DF(1, 1, 0, 0.15f), Vector4DF(1, 1, 1, 0.2f));
		gvdb.CommitTransferFunc();
		//Render
		cudaDeviceSynchronize();
		auto start = std::chrono::high_resolution_clock::now();
		gvdb.Render(SHADE_VOLUME, 0, 0);
		cudaDeviceSynchronize();
		auto finish = std::chrono::high_resolution_clock::now();
		if (secondsOut) *secondsOut = std::chrono::duration<double>(finish - start).count();
	}
	//grab output
	if (targetDevice != 0)
	{
		if (cudaDeviceSynchronize() != cudaSuccess)
			std::cout << "Error before memcpy" << std::endl;
		std::cout << "source pointer: 0x" << std::hex << gvdb.mRenderBuf[0].gpu
			<< " (" << getPointerType(gvdb.mRenderBuf[0].gpu) << ")" << std::endl;
		std::cout << "target pointer: 0x" << std::hex << targetDevice
			<< " (" << getPointerType(targetDevice) << ")" << std::endl;
		size_t num_bytes = args.resolutionX * args.resolutionY * 12 * sizeof(float);
		cudaError_t err = cudaMemcpy(reinterpret_cast<void*>(targetDevice),
			reinterpret_cast<const void*>(gvdb.mRenderBuf[0].gpu),
			num_bytes, cudaMemcpyDeviceToDevice);
		if (err == cudaErrorInvalidValue)
			std::cout << "Invalid value" << std::endl;
		//cudaCheck(
		//	cuMemcpyDtoD(targetDevice, gvdb.mRenderBuf[0].gpu num_bytes),
		//	"cuMemcpyDtoD");
		cudaCheck(cuCtxSynchronize(), "cudaDeviceSynchronize()");
		return {};
	}

	std::vector<float> buf(args.resolutionX * args.resolutionY * 12);
	gvdb.ReadRenderBuf(0, reinterpret_cast<uchar*>(&buf[0]));

	//screen space ambient occlusion
	if (args.aoMode == Args::AOScreen)
	{
		ShadowFX::screenSpaceAmbientOcclusion(
			args.resolutionX, args.resolutionY,
			buf.data(), 12, args.resolutionX, 4, 5, 6, 7,
			&buf[0], 12, args.resolutionX, 10,
			64, args.aoRadius);
	}

	return buf;
}

void saveImage(const std::vector<float>& image,
	int width, int height,
	const std::string& outputFilename, 
	int startChannel, int numChannels)
{
	std::cout << "Save Image to " << outputFilename << std::endl;
	if (boost::iends_with(outputFilename, ".exr")) {
		// Save as EXR (slow, but small file size).
		Imf::setGlobalThreadCount(8);
		Imf::Header header(width, height);
		header.compression() = Imf::ZIP_COMPRESSION;
		header.channels().insert("R", Imf::Channel(Imf::FLOAT));
		header.channels().insert("G", Imf::Channel(Imf::FLOAT));
		header.channels().insert("B", Imf::Channel(Imf::FLOAT));
		header.channels().insert("A", Imf::Channel(Imf::FLOAT));
		const size_t pixelBytes = sizeof(float) * 12;
		const size_t rowBytes = pixelBytes * width;
		Imf::FrameBuffer framebuffer;
		float* data = const_cast<float*>(image.data());
		float dummyZero = 0.0f;
		framebuffer.insert("R",
			Imf::Slice(Imf::FLOAT, reinterpret_cast<char*>(data + startChannel), pixelBytes, rowBytes));
		if (numChannels >= 2)
			framebuffer.insert("G",
				Imf::Slice(Imf::FLOAT, reinterpret_cast<char*>(data + startChannel + 1), pixelBytes, rowBytes));
		//else
		//	framebuffer.insert("G", Imf::Slice(Imf::FLOAT, reinterpret_cast<char*>(&dummyZero)));
		if (numChannels >= 3)
			framebuffer.insert("B",
				Imf::Slice(Imf::FLOAT, reinterpret_cast<char*>(data + startChannel + 2), pixelBytes, rowBytes));
		//else
		//	framebuffer.insert("B", Imf::Slice(Imf::FLOAT, reinterpret_cast<char*>(&dummyZero)));
		if (numChannels >= 4)
			framebuffer.insert("A",
				Imf::Slice(Imf::FLOAT, reinterpret_cast<char*>(data + startChannel + 3), pixelBytes, rowBytes));
		//else
		//	framebuffer.insert("A", Imf::Slice(Imf::FLOAT, reinterpret_cast<char*>(&dummyZero)));

		Imf::OutputFile imgFile(outputFilename.c_str(), header);
		imgFile.setFrameBuffer(framebuffer);
		imgFile.writePixels(height);
	}
	else {
		std::cout << "unsupported image file format (" << outputFilename << ")" << std::endl;
	}
}

void renderSingle(const Args& args)
{
	//high-res
	auto image = render(args,
		args.cameraOrigin, args.cameraLookAt,
		args.cameraOrigin, args.cameraLookAt);
	saveImage(image, args.resolutionX, args.resolutionY, args.outputFilename, 0, 4);
	if (!args.highDepthFilename.empty())
		saveImage(image, args.resolutionX, args.resolutionY, args.highDepthFilename, 4, 4);
	if (!args.highEffectsFilename.empty())
		saveImage(image, args.resolutionX, args.resolutionY, args.highEffectsFilename, 10, 1);

	//low-res
	if (!args.downscaleFilename.empty())
	{
		Args args2 = args;
		args2.samples = 1;
		args2.resolutionX = args.resolutionX / args.downscaleFactor;
		args2.resolutionY = args.resolutionY / args.downscaleFactor;
		auto images = render(args2,
			args.cameraOrigin, args.cameraLookAt,
			args.cameraOrigin, args.cameraLookAt);
		saveImage(images, args2.resolutionX, args2.resolutionY, args.downscaleFilename, 0, 4);
		if (!args.downscaleDepthFilename.empty())
			saveImage(images, args2.resolutionX, args2.resolutionY, args.downscaleDepthFilename, 4, 4);
	}
}

Vector3DF interpolate(const Vector3DF& a, const Vector3DF& b, float alpha)
{
	return Vector3DF(a) * (1 - alpha) + Vector3DF(b) * alpha;
}
void renderAnimation(const Args& args)
{
	for (int frame = 0; frame < args.animFrames; ++frame)
	{
		Vector3DF currentCameraOrigin = interpolate(
			args.cameraOrigin, args.cameraOriginEnd,
			frame / double(args.animFrames - 1));
		Vector3DF nextCameraOrigin = interpolate(
			args.cameraOrigin, args.cameraOriginEnd,
			(frame + 1) / double(args.animFrames - 1));
		Vector3DF currentCameraLookAt = interpolate(
			args.cameraLookAt, args.cameraLookAtEnd,
			frame / double(args.animFrames - 1));
		Vector3DF nextCameraLookAt = interpolate(
			args.cameraLookAt, args.cameraLookAtEnd,
			(frame + 1) / double(args.animFrames - 1));

		//high-res
		float time;
		auto images = render(args,
			currentCameraOrigin, currentCameraLookAt,
			nextCameraOrigin, nextCameraLookAt,
			&time);
		saveImage(images, args.resolutionX, args.resolutionY, tfm::format(args.outputFilename, frame), 0, 4);
		if (!args.highDepthFilename.empty())
			saveImage(images, args.resolutionX, args.resolutionY, tfm::format(args.highDepthFilename, frame), 4, 4);
		if (!args.highEffectsFilename.empty())
			saveImage(images, args.resolutionX, args.resolutionY, tfm::format(args.highEffectsFilename, frame), 10, 1);
		std::cout << "Time to render high resolution: " << time << " sec" << std::endl;

		//low-res
		if (!args.downscaleFilename.empty())
		{
			Args args2 = args;
			args2.samples = 1;
			args2.resolutionX = args.resolutionX / args.downscaleFactor;
			args2.resolutionY = args.resolutionY / args.downscaleFactor;
			auto images = render(args2,
				currentCameraOrigin, currentCameraLookAt,
				nextCameraOrigin, nextCameraLookAt,
				&time);
			saveImage(images, args2.resolutionX, args2.resolutionY, tfm::format(args.downscaleFilename, frame), 0, 4);
			if (!args.downscaleDepthFilename.empty())
				saveImage(images, args2.resolutionX, args2.resolutionY, tfm::format(args.downscaleDepthFilename, frame), 4, 4);
			if (!args.downscaleFlowFilename.empty())
				saveImage(images, args2.resolutionX, args2.resolutionY, tfm::format(args.downscaleFlowFilename, frame), 8, 2);
			std::cout << "Time to render low resolution: " << time << " sec" << std::endl;
		}
	}
}

void renderPipe(Args args)
{
	std::cout << "Enter Pipe mode and wait for commands" << std::endl;

	Vector3DF lastOrigin = args.cameraOrigin;
	Vector3DF lastLookAt = args.cameraLookAt;

	if (args.pipeOutputFormat != "RGBMXYZDUV")
	{
		std::cout << "Unsupported pipe output format '" << args.pipeOutputFormat
			<< "', only 'RGBMXYZDUV' is supported" << std::endl;
		return;
	}
#define IDX_OUT(c, y, x) ((x) + args.resolutionX * ((y) + args.resolutionY * (c)))
#define IDX_IN(c, y, x) ((c) + 12 * ((x) + args.resolutionX * (y)))

	static const int32_t scale = 1000;

	while (true)
	{
		std::string cmd, value;
		std::getline(std::cin, cmd);
		auto idx = cmd.find('=');
		if (idx != std::string::npos)
		{
			value = cmd.substr(idx + 1);
			cmd = cmd.substr(0, idx);
		}
		if (cmd == "exit")
		{
			std::cout << "Exit program" << std::endl;
			return;
		}
		else if (cmd == "render")
		{
			//render
			float time;
			auto images = render(args,
				args.cameraOrigin, args.cameraLookAt,
				lastOrigin, lastLookAt, &time); //use last camera as next camera and invert flow
			//write to stdout (cerr)
			std::vector<float> output(args.resolutionX * args.resolutionY * 12 + 1, 0.5f);
			#pragma omp parallel for
			for (int y = 0; y < args.resolutionY; ++y)
				for (int x = 0; x < args.resolutionX; ++x)
					for (int c = 0; c < 12; ++c)
						output[IDX_OUT(c, y, x)] = images[IDX_IN(c, y, x)];
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
		}
		else if (cmd == "renderDirect")
		{
			//render directly to target memory.
			//Output format: H x W x C
			CUdeviceptr targetHost;
			parseStr(value, targetHost);
			float time;
			std::cout << "Render to " << targetHost << std::endl;
			render(args,
				args.cameraOrigin, args.cameraLookAt,
				lastOrigin, lastLookAt, &time, targetHost);
			//write time
			std::cerr.write(reinterpret_cast<const char*>(&time), sizeof(float));
			std::cerr.flush();
			//save old camera position for flow
			lastOrigin = args.cameraOrigin;
			lastLookAt = args.cameraLookAt;
		}
		else
		{ //command=value
			if (cmd == "cameraOrigin")
				parseStr(value, args.cameraOrigin.x, args.cameraOrigin.y, args.cameraOrigin.z);
			else if (cmd == "cameraLookAt")
				parseStr(value, args.cameraLookAt.x, args.cameraLookAt.y, args.cameraLookAt.z);
			else if (cmd == "cameraUp")
				parseStr(value, args.cameraUp.x, args.cameraUp.y, args.cameraUp.z);
			else if (cmd == "cameraFoV")
				parseStr(value, args.cameraFov);
			else if (cmd == "resolution") {
				parseStr(value, args.resolutionX, args.resolutionY);
				args.viewport = make_int4(0, 0, args.resolutionX, args.resolutionY);
			}
			else if (cmd == "isovalue")
				parseStr(value, args.isovalue);
			else if (cmd == "unshaded")
				parseStr(value, args.noShading);
			else if (cmd == "aosamples")
				parseStr(value, args.aoSamples);
			else if (cmd == "aoradius")
				parseStr(value, args.aoRadius);
			else if (cmd == "viewport") //minX, minY, maxX, maxY
				parseStr(value, args.viewport.x, args.viewport.y, args.viewport.z, args.viewport.w);
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

	if (args.renderMode == Args::RenderModeConvert) {
		convert(args);
		return 0;
	}

	initGVDB();
	loadGrid(args);
	initRendering(args);
	std::cout << "Done with the initialization" << std::endl;

	if (args.outputFilename == "PIPE")
		renderPipe(args);
	else if (args.animFrames > 1)
		renderAnimation(args);
	else
		renderSingle(args);
}
