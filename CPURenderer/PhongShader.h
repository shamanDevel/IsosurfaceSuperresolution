#pragma once

#define _USE_MATH_DEFINES
#include <math.h>
#include <openvdb/tools/RayTracer.h>

template <typename SamplerType>
class PhongShader : public openvdb::tools::BaseShader
{
public:
	PhongShader(
		const openvdb::Vec3R lightDir,
		const openvdb::Vec3R ambient,
		const openvdb::Vec3R diffuse,
		const openvdb::Vec3R specular,
		const int specularExponent,
		const double alpha = 1.0)
		: lightDir(lightDir)
		, ambient(ambient)
		, diffuse(diffuse)
		, specular(specular)
		, specularExponent(specularExponent)
		, alpha(alpha)
	{}
	PhongShader(const PhongShader&) = default;
	~PhongShader() override = default;
	openvdb::tools::Film::RGBA operator()(
		const openvdb::Vec3R& pos, const openvdb::Vec3R& normal, const openvdb::Vec3R& rayDir) const override
	{
		//two-sided phong shading
		openvdb::Vec3R col = ambient;
		col += diffuse * openvdb::math::Abs(normal.dot(lightDir));
		openvdb::Vec3R reflect = 2 * (normal.dot(lightDir))*normal - lightDir;
		col += specular * ((specularExponent + 2) / (2 * M_PI))
		* openvdb::math::Pow(openvdb::math::Max(0.0, reflect.dot(rayDir)), specularExponent);

		return openvdb::tools::Film::RGBA(col.x(), col.y(), col.z(), alpha);
	}
	BaseShader* copy() const override { return new PhongShader<SamplerType>(*this); }

private:
	const openvdb::Vec3R lightDir;
	const openvdb::Vec3R ambient;
	const openvdb::Vec3R diffuse;
	const openvdb::Vec3R specular;
	const int specularExponent;
	const double alpha;
};
