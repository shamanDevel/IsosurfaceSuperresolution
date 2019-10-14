#pragma once

#include <openvdb/Types.h>
#include <openvdb/math/BBox.h>
#include <openvdb/math/Ray.h>
#include <openvdb/math/Math.h>
#include <openvdb/tools/RayIntersector.h>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/tools/RayTracer.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace math
{
	//////////////////////////////////////////// VolumeHDDA /////////////////////////////////////////////

	/// @brief Helper class that implements ray - isosurface intersection in generic volumes.
	///
	/// @details The template argument ChildNodeLevel specifies the entry
	/// upper node level used for the hierarchical ray-marching. The final
	/// lowest level is always the leaf node level, i.e. not the voxel level!
	template <typename TreeT, typename RayT, typename StencilT, int ChildNodeLevel>
	class IsoVolumeHDDA
	{
		friend class IsoVolumeHDDA<TreeT, RayT, StencilT, ChildNodeLevel + 1>;
	public:

		using ChainT = typename TreeT::RootNodeType::NodeChainType;
		using NodeT = typename boost::mpl::at<ChainT, boost::mpl::int_<ChildNodeLevel> >::type;
		using TimeSpanT = typename RayT::TimeSpan;

		IsoVolumeHDDA() {}

		template <typename AccessorT>
		bool hits(RayT& ray, AccessorT &acc, StencilT &stencil, double isovalue, double& time)
		{
			mDDA.init(ray);
			do {
				if (acc.template probeConstNode<NodeT>(mDDA.voxel()) != nullptr) {//child node
					ray.setTimes(mDDA.time(), mDDA.next());
					if (mHDDA.hits(ray, acc, stencil, isovalue, time)) return true;
				}
			} while (mDDA.step());
			return false;
		}

	private:

		math::DDA<RayT, NodeT::TOTAL> mDDA;
		IsoVolumeHDDA<TreeT, RayT, StencilT, ChildNodeLevel - 1> mHDDA;
	};

	/// @brief Specialization of Hierarchical Digital Differential Analyzer
	/// class that intersects against the leafs or tiles of a generic volume.
	template <typename TreeT, typename RayT, typename StencilT>
	class IsoVolumeHDDA<TreeT, RayT, StencilT, -1>
	{
	public:

		IsoVolumeHDDA() {}
		using ValueT = typename StencilT::ValueType;

	private:
		inline ValueT interpValue(RayT& ray, StencilT &stencil, double isovalue, double time)
		{
			const auto pos = ray(time);
			stencil.moveTo(pos);
			return stencil.interpolation(pos) - isovalue;
		}

		static inline double interpTime(const ValueT& v0, const ValueT& v1, const double& t0, const double& t1)
		{
			assert(math::isApproxLarger(t1, t0, double(1e-6)));
			return t0 + (t1 - t0)*v0 / (v0 - v1);
		}

	public:
		template <typename AccessorT>
		bool hits(RayT& ray, AccessorT &acc, StencilT &stencil, double isovalue, double& time)
		{
			mDDA.init(ray);
			double t0 = mDDA.time();
			ValueT v0 = interpValue(ray, stencil, isovalue, t0);
			do {
				const auto ijk = mDDA.voxel();
				double t1 = mDDA.next();
				ValueT V;
				//if (stencil.accessor().probeValue(ijk, V)) {
					ValueT v1 = this->interpValue(ray, stencil, isovalue, t1);
					if (math::ZeroCrossing(v0, v1)) {
						static const int BINARY_STEPS = 5;
						time = 0.5 * (t0 + t1);
						for (int i=0; i<BINARY_STEPS; ++i)
						{
							ValueT v2 = this->interpValue(ray, stencil, isovalue, time);
							if (math::ZeroCrossing(v0, v2)) {
								t1 = time;
							} else {
								t0 = time;
								v0 = v2;
							}
							time = 0.5 * (t0 + t1);
						}
						//time = interpTime(v0, v1, t0, t1);
						return true;
					}
					t0 = t1;
					v0 = v1;
				//}
			} while (mDDA.step());
			return false;
		}

	private:
		math::DDA<RayT, 0> mDDA;
	};

}

namespace tools {

////////////////////////////////////// IsoVolumeRayIntersector //////////////////////////////////////


/// @brief This class provides the public API for intersecting a ray
/// with an isosurface inside a generic (e.g. density) volume.
/// @details Internally it performs the actual hierarchical tree node traversal.
/// @warning Use the (default) copy-constructor to make sure each
/// computational thread has their own instance of this class. This is
/// important since it contains a ValueAccessor that is
/// not thread-safe and a CoordBBox of the active voxels that should
/// not be re-computed for each thread. However copying is very efficient.
/// @par Example:
/// @code
/// // Create an instance for the master thread
/// VolumeRayIntersector inter(grid);
/// // For each additional thread use the copy constructor. This
/// // amortizes the overhead of computing the bbox of the active voxels!
/// VolumeRayIntersector inter2(inter);
/// // Before each ray-traversal set the index ray.
/// iter.setIndexRay(ray);
/// // or world ray
/// iter.setWorldRay(ray);
/// // Now you can begin the ray-marching using consecutive calls to VolumeRayIntersector::march
/// double t0=0, t1=0;// note the entry and exit times are with respect to the INDEX ray
/// while ( inter.march(t0, t1) ) {
///   // perform line-integration between t0 and t1
/// }}
/// @endcode
template<typename GridT,
	int NodeLevel = GridT::TreeType::RootNodeType::ChildNodeType::LEVEL,
	typename RayT = math::Ray<Real>,
	typename StencilT = math::BoxStencil<GridT>>
	class IsoVolumeRayIntersector
{
public:
	using GridType = GridT;
	using RayType = RayT;
	using RealType = typename RayT::RealType;
	using RootType = typename GridT::TreeType::RootNodeType;
	using TreeT = tree::Tree<typename RootType::template ValueConverter<bool>::Type>;
	using Vec3Type = Vec3R;

	static_assert(NodeLevel >= 0 && NodeLevel < int(TreeT::DEPTH) - 1, "NodeLevel out of range");

	/// @brief Grid constructor
	/// @param grid Generic grid to intersect rays against.
	/// @param dilationCount The number of voxel dilations performed
	/// on (a boolean copy of) the input grid. This allows the
	/// intersector to account for the size of interpolation kernels
	/// in client code.
	/// @throw RuntimeError if the voxels of the grid are not uniform
	/// or the grid is empty.
	IsoVolumeRayIntersector(const GridT& grid, double isovalue, int dilationCount = 0)
		: mIsMaster(true)
		, mTree(new TreeT(grid.tree(), false, TopologyCopy()))
		, mGrid(&grid)
		, mAccessor(*mTree)
		, mIsovalue(isovalue)
		, mStencil(grid)
	{
		if (!grid.hasUniformVoxels()) {
			OPENVDB_THROW(RuntimeError,
				"IsoVolumeRayIntersector only supports uniform voxels!");
		}
		if (grid.empty()) {
			OPENVDB_THROW(RuntimeError, "LinearSearchImpl does not supports empty grids");
		}

		// Dilate active voxels to better account for the size of interpolation kernels
		tools::dilateVoxels(*mTree, dilationCount);

		mTree->root().evalActiveBoundingBox(mBBox, /*visit individual voxels*/false);

		mBBox.max().offset(1);//padding so the bbox of a node becomes (origin,origin + node_dim)
	}

	/// @brief Grid and BBox constructor
	/// @param grid Generic grid to intersect rays against.
	/// @param bbox The axis-aligned bounding-box in the index space of the grid.
	/// @warning It is assumed that bbox = (min, min + dim) where min denotes
	/// to the smallest grid coordinates and dim are the integer length of the bbox.
	/// @throw RuntimeError if the voxels of the grid are not uniform
	/// or the grid is empty.
	IsoVolumeRayIntersector(const GridT& grid, double isovalue, const math::CoordBBox& bbox)
		: mIsMaster(true)
		, mTree(new TreeT(grid.tree(), false, TopologyCopy()))
		, mGrid(&grid)
		, mAccessor(*mTree)
		, mBBox(bbox)
		, mIsovalue(isovalue)
		, mStencil(grid)
	{
		if (!grid.hasUniformVoxels()) {
			OPENVDB_THROW(RuntimeError,
				"VolumeRayIntersector only supports uniform voxels!");
		}
		if (grid.empty()) {
			OPENVDB_THROW(RuntimeError, "LinearSearchImpl does not supports empty grids");
		}
	}

	/// @brief Shallow copy constructor
	/// @warning This copy constructor creates shallow copies of data
	/// members of the instance passed as the argument. For
	/// performance reasons we are not using shared pointers (their
	/// mutex-lock impairs multi-threading).
	IsoVolumeRayIntersector(const IsoVolumeRayIntersector& other)
		: mIsMaster(false)
		, mTree(other.mTree)//shallow copy
		, mGrid(other.mGrid)//shallow copy
		, mAccessor(*mTree)//initialize new (vs deep copy)
		, mTmax(other.mTmax)//deep copy
		, mBBox(other.mBBox)//deep copy
		, mIsovalue(other.mIsovalue)
		, mStencil(other.mStencil)
	{
	}

	/// @brief Destructor
	~IsoVolumeRayIntersector() { if (mIsMaster) delete mTree; }

	/// @brief Return @c true if the index-space ray intersects the level set
	/// @param iRay  ray represented in index space.
	/// @param iTime if an intersection was found it is assigned the time of the
	///              intersection along the index ray.
	bool intersectsIS(RayType iRay, RealType &iTime, Vec3R& iPos)
	{
		if (!iRay.clip(mBBox)) return false;
		bool ret = mHDDA.hits(iRay, mAccessor, mStencil, mIsovalue, iTime);
		if (ret) iPos = iRay(iTime);
		return ret;
	}

	/// @brief Return @c true if the world-space ray intersects the level set.
	/// @param wRay   ray represented in world space.
	/// @param wTime  if an intersection was found it is assigned the time of the
	///               intersection along the world ray.
	bool intersectsWS(const RayType& wRay, RealType &wTime, Vec3R& wPos)
	{
		RayType iRay = wRay.worldToIndex(*mGrid);
		if (!iRay.clip(mBBox)) return false;
		double iTime;
		bool ret = mHDDA.hits(iRay, mAccessor, mStencil, mIsovalue, iTime);
		if (ret) {
			wTime = iTime * mGrid->transform().baseMap()->applyJacobian(iRay.dir()).length();
			wPos = mGrid->indexToWorld(iRay(iTime));
		}
		return ret;
	}

	Vec3R gradient(Vec3R iPos)
	{
		//mStencil.gradient(iPos) returns the derivative of trilinear interpolation, which is constant!
		// Hence, I have blocking artifacts
		//Instead, I compute the numerical difference now
		static const float delta = 1.f;
		Vec3R normal;
		//x
		mStencil.moveTo(iPos + Vec3R(delta, 0, 0)); normal.x() = mStencil.interpolation(iPos + Vec3R(delta, 0, 0));
		mStencil.moveTo(iPos - Vec3R(delta, 0, 0)); normal.x() -= mStencil.interpolation(iPos - Vec3R(delta, 0, 0));
		//y
		mStencil.moveTo(iPos + Vec3R(0, delta, 0)); normal.y() = mStencil.interpolation(iPos + Vec3R(0, delta, 0));
		mStencil.moveTo(iPos - Vec3R(0, delta, 0)); normal.y() -= mStencil.interpolation(iPos - Vec3R(0, delta, 0));
		//z
		mStencil.moveTo(iPos + Vec3R(0, 0, delta)); normal.z() = mStencil.interpolation(iPos + Vec3R(0, 0, delta));
		mStencil.moveTo(iPos - Vec3R(0, 0, delta)); normal.z() -= mStencil.interpolation(iPos - Vec3R(0, 0, delta));

		return normal;
	}

	bool intersectsWS(const RayType& wRay, double& wTime, Vec3R& world, Vec3R& normal)
	{
		RayType iRay = wRay.worldToIndex(*mGrid);
		if (!iRay.clip(mBBox)) return false;
		double iTime;
		bool ret = mHDDA.hits(iRay, mAccessor, mStencil, mIsovalue, iTime);
		if (ret) {
			Vec3R iPos = iRay(iTime);
			world = mGrid->indexToWorld(iPos);
			mStencil.moveTo(iPos);
			normal = gradient(iPos);
			normal.normalize();
			wTime = iTime * mGrid->transform().baseMap()->applyJacobian(iRay.dir()).length();
		}
		return ret;
	}

	/// @brief Return a const reference to the input grid.
	const GridT& grid() const { return *mGrid; }

	/// @brief Return a const reference to the (potentially dilated)
	/// bool tree used to accelerate the ray marching.
	const TreeT& tree() const { return *mTree; }

	/// @brief Return a const reference to the BBOX of the grid
	const math::CoordBBox& bbox() const { return mBBox; }

	/// @brief Print bbox, statistics, memory usage and other information.
	/// @param os            a stream to which to write textual information
	/// @param verboseLevel  1: print bbox only; 2: include boolean tree
	///                      statistics; 3: include memory usage
	void print(std::ostream& os = std::cout, int verboseLevel = 1)
	{
		if (verboseLevel > 0) {
			os << "BBox: " << mBBox << std::endl;
			if (verboseLevel == 2) {
				mTree->print(os, 1);
			}
			else if (verboseLevel > 2) {
				mTree->print(os, 2);
			}
		}
	}

private:
	using AccessorT = typename tree::ValueAccessor<const TreeT,/*IsSafe=*/false>;

	math::IsoVolumeHDDA<TreeT, RayT, StencilT, NodeLevel> mHDDA;
	const bool      mIsMaster;
	TreeT*          mTree;
	const GridT*    mGrid;
	AccessorT       mAccessor;
	RealType        mTmax;
	math::CoordBBox mBBox;
	double          mIsovalue;
	StencilT        mStencil;

};// VolumeRayIntersector

/////////////////////////////// ISO VOLUME RAY TRACER ///////////////////////////////////////

/// @brief A (very) simple multithreaded ray tracer specifically for narrow-band level sets.
/// @details Included primarily as a reference implementation.
template<typename GridT, typename IntersectorT = tools::IsoVolumeRayIntersector<GridT> >
class IsoVolumeRayTracer
{
public:
	using GridType = GridT;
	using Vec3Type = typename IntersectorT::Vec3Type;
	using RayType = typename IntersectorT::RayType;

	/// @brief Constructor based on an instance of the grid to be rendered.
	IsoVolumeRayTracer(const GridT& grid,
		const BaseShader& shader,
		BaseCamera& camera,
		size_t pixelSamples = 1,
		unsigned int seed = 0)
		: mIsMaster(true),
		mRand(nullptr),
		mInter(grid),
		mShader(shader.copy()),
		mCamera(&camera),
		mDepthNormalFilm(nullptr),
		mFlowFilm(nullptr),
		mNextCamera(nullptr)
	{
		this->setPixelSamples(pixelSamples, seed);
	}

	/// @brief Constructor based on an instance of the intersector
	/// performing the ray-intersections.
	IsoVolumeRayTracer(const IntersectorT& inter,
		const BaseShader& shader,
		BaseCamera& camera,
		size_t pixelSamples = 1,
		unsigned int seed = 0)
		: mIsMaster(true),
		mRand(nullptr),
		mInter(inter),
		mShader(shader.copy()),
		mCamera(&camera),
		mDepthNormalFilm(nullptr),
		mFlowFilm(nullptr),
		mNextCamera(nullptr)
	{
		this->setPixelSamples(pixelSamples, seed);
	}

	/// @brief Copy constructor
	IsoVolumeRayTracer(const IsoVolumeRayTracer& other) :
		mIsMaster(false),
		mRand(other.mRand),
		mInter(other.mInter),
		mShader(other.mShader->copy()),
		mCamera(other.mCamera),
		mSubPixels(other.mSubPixels),
		mDepthNormalFilm(other.mDepthNormalFilm),
		mFlowFilm(other.mFlowFilm),
		mNextCamera(other.mNextCamera)
	{
	}

	/// @brief Destructor
	~IsoVolumeRayTracer()
	{
		if (mIsMaster) delete[] mRand;
	}

	/// @brief Set the level set grid to be ray-traced
	void setGrid(const GridT& grid)
	{
		assert(mIsMaster);
		mInter = IntersectorT(grid);
	}

	/// @brief Set the intersector that performs the actual
	/// intersection of the rays against the narrow-band level set.
	void setIntersector(const IntersectorT& inter)
	{
		assert(mIsMaster);
		mInter = inter;
	}

	/// @brief Set the shader derived from the abstract BaseShader class.
	///
	/// @note The shader is not assumed to be thread-safe so each
	/// thread will get its only deep copy. For instance it could
	/// contains a ValueAccessor into another grid with auxiliary
	/// shading information. Thus, make sure it is relatively
	/// light-weight and efficient to copy (which is the case for ValueAccesors).
	void setShader(const BaseShader& shader)
	{
		assert(mIsMaster);
		mShader.reset(shader.copy());
	}

	/// @brief Set the camera derived from the abstract BaseCamera class.
	void setCamera(BaseCamera& camera)
	{
		assert(mIsMaster);
		mCamera = &camera;
	}


	/// @brief Set the number of pixel samples and the seed for
	/// jittered sub-rays. A value larger than one implies
	/// anti-aliasing by jittered super-sampling.
	/// @throw ValueError if pixelSamples is equal to zero.
	void setPixelSamples(size_t pixelSamples, unsigned int seed = 0)
	{
		assert(mIsMaster);
		if (pixelSamples == 0) {
			OPENVDB_THROW(ValueError, "pixelSamples must be larger than zero!");
		}
		mSubPixels = pixelSamples - 1;
		delete[] mRand;
		if (mSubPixels > 0) {
			mRand = new double[16];
			math::Rand01<double> rand(seed);//offsets for anti-aliaing by jittered super-sampling
			for (size_t i = 0; i < 16; ++i) mRand[i] = rand();
		}
		else {
			mRand = nullptr;
		}
	}

	/// @brief Sets the film for depth and normal.
	/// RGB: xyz of the normal, alpha: depth value
	void setDepthNormalFilm(Film* film)
	{
		this->mDepthNormalFilm = film;
	}

	void setFlowFilm(Film* film, BaseCamera* nextCamera)
	{
		this->mFlowFilm = film;
		this->mNextCamera = nextCamera;
	}

	/// @brief Perform the actual (potentially multithreaded) ray-tracing.
	void render(bool threaded = true) const
	{
		tbb::blocked_range<size_t> range(0, mCamera->height());
		threaded ? tbb::parallel_for(range, *this) : (*this)(range);
	}

	/// @brief Public method required by tbb::parallel_for.
	/// @warning Never call it directly.
	void operator()(const tbb::blocked_range<size_t>& range) const
	{
		//compute normal matrix
		auto cameraMatrix = mCamera->getAffineMap().getConstMat4();
		auto viewMatrix = cameraMatrix.inverse();
		auto normalMatrix = viewMatrix.getMat3();
		auto nextViewMatrix = mNextCamera
			? mNextCamera->getAffineMap().getConstMat4().inverse()
			: viewMatrix;

		const BaseShader& shader = *mShader;
		IntersectorT inter(mInter);
		Vec3Type xyz, nml, xyz2, nml2;
		double wTime;
		const float frac = 1.0f / (1.0f + mSubPixels);
		for (size_t j = range.begin(), n = 0, je = range.end(); j < je; ++j) {
			for (size_t i = 0, ie = mCamera->width(); i < ie; ++i) {
				Film::RGBA& bg = mCamera->pixel(i, j);
				wTime = 0;
				nml.setZero();
				RayType ray = mCamera->getRay(i, j);//primary ray
				Film::RGBA c = inter.intersectsWS(ray, wTime, xyz, nml) ? shader(xyz, nml, ray.dir()) : bg;
				for (size_t k = 0; k < mSubPixels; ++k, n += 2) {
					ray = mCamera->getRay(i, j, mRand[n & 15], mRand[(n + 1) & 15]);
					c += inter.intersectsWS(ray, wTime, xyz2, nml2) ? shader(xyz2, nml2, ray.dir()) : bg;
				}//loop over sub-pixels
				bg = c * frac;
				if (wTime == 0) bg.a = 0;
				//write normal and depth
				if (mDepthNormalFilm && wTime>0)
				{
					auto nml2 = nml * normalMatrix; //transform world depth to view depth
					if (nml2.z() < 0) nml2 = -nml2;
					mDepthNormalFilm->pixel(i, j) = Film::RGBA(nml2.x(), nml2.y(), nml2.z(), wTime);
				}
				//write flow
				if (mFlowFilm && mNextCamera && wTime>0)
				{
					Vec4d xyzw(xyz.x(), xyz.y(), xyz.z(), 1.0);
					Vec4d screenCurrent = xyzw * viewMatrix;
					Vec4d screenNext = xyzw * nextViewMatrix;
					screenCurrent /= screenCurrent.w();
					screenNext /= screenNext.w();
					double velX = screenNext.x() - screenCurrent.x();
					double velY = screenNext.y() - screenCurrent.y();
					mFlowFilm->pixel(i, j) = Film::RGBA(velX, velY, 0);
				}
			}//loop over image height
		}//loop over image width
	}

private:
	const bool                          mIsMaster;
	double*                             mRand;
	IntersectorT                        mInter;
	std::unique_ptr<const BaseShader>   mShader;
	BaseCamera*                         mCamera;
	size_t                              mSubPixels;
	Film*                               mDepthNormalFilm;
	BaseCamera*                         mNextCamera;
	Film*                               mFlowFilm;
};// LevelSetRayTracer

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb
