#include "pch.h"
#include "Vdb2Vbx.h"

#include <iostream>
#include <gvdb/gvdb_allocator.h>

using namespace nvdb;

namespace my
{
	class OVDBGrid {
	public:

		FloatGrid543::Ptr			grid543F;			// grids
		Vec3fGrid543::Ptr			grid543VF;
		FloatGrid34::Ptr			grid34F;
		Vec3fGrid34::Ptr			grid34VF;

		TreeType543F::LeafCIter		iter543F;			// iterators
		TreeType543VF::LeafCIter	iter543VF;
		TreeType34F::LeafCIter		iter34F;
		TreeType34VF::LeafCIter		iter34VF;
		// buffers
		openvdb::tree::LeafNode<float, 3U>::Buffer buf3U;	// 2^3 leaf res
		openvdb::tree::LeafNode<Vec3f, 3U>::Buffer buf3VU;
		openvdb::tree::LeafNode<float, 4U>::Buffer buf4U;	// 2^4 leaf res
		openvdb::tree::LeafNode<Vec3f, 4U>::Buffer buf4VU;
	};

}

static void vdbSkip(my::OVDBGrid* ovg, int leaf_start, int gt, bool isFloat)
{
	switch (gt) {
	case 0:
		if (isFloat) { ovg->iter543F = ovg->grid543F->tree().cbeginLeaf();  for (int j = 0; ovg->iter543F && j < leaf_start; j++) ++ovg->iter543F; }
		else { ovg->iter543VF = ovg->grid543VF->tree().cbeginLeaf(); for (int j = 0; ovg->iter543VF && j < leaf_start; j++) ++ovg->iter543VF; }
		break;
	case 1:
		if (isFloat) { ovg->iter34F = ovg->grid34F->tree().cbeginLeaf();  for (int j = 0; ovg->iter34F && j < leaf_start; j++) ++ovg->iter34F; }
		else { ovg->iter34VF = ovg->grid34VF->tree().cbeginLeaf(); for (int j = 0; ovg->iter34VF && j < leaf_start; j++) ++ovg->iter34VF; }
		break;
	};
}

static bool vdbCheck(my::OVDBGrid* ovg, int gt, bool isFloat)
{
	switch (gt) {
	case 0: return (isFloat ? ovg->iter543F.test() : ovg->iter543VF.test());	break;
	case 1: return (isFloat ? ovg->iter34F.test() : ovg->iter34VF.test());	break;
	};
	return false;
}
static void vdbOrigin(my::OVDBGrid* ovg, Coord& orig, int gt, bool isFloat)
{
	switch (gt) {
	case 0: if (isFloat) (ovg->iter543F)->getOrigin(orig); else (ovg->iter543VF)->getOrigin(orig);	break;
	case 1: if (isFloat) (ovg->iter34F)->getOrigin(orig); else (ovg->iter34VF)->getOrigin(orig);	break;
	};
}

static void vdbNext(my::OVDBGrid* ovg, int gt, bool isFloat)
{
	switch (gt) {
	case 0: if (isFloat) ovg->iter543F.next();	else ovg->iter543VF.next();		break;
	case 1: if (isFloat) ovg->iter34F.next();	else ovg->iter34VF.next();		break;
	};
}

bool Vdb2Vbx::LoadVDB(nvdb::VolumeGVDB& gvdb, const std::string& filename)
{
	static const int APRON = 1;

	openvdb::initialize();
	FloatGrid34::registerGrid();
	//FloatGridVF34::registerGrid();

	my::OVDBGrid* mOVDB = new my::OVDBGrid;

	CoordBBox box;
	Coord orig;
	Vector3DF p0, p1;

	PERF_PUSH("Clear grid");

	if (mOVDB->grid543F != 0x0) 	mOVDB->grid543F.reset();
	if (mOVDB->grid543VF != 0x0) 	mOVDB->grid543VF.reset();
	if (mOVDB->grid34F != 0x0) 	mOVDB->grid34F.reset();
	if (mOVDB->grid34VF != 0x0) 	mOVDB->grid34VF.reset();

	PERF_POP();

	PERF_PUSH("Load VDB");

	// Read .vdb file	

	std::cout << "   Reading OpenVDB file.\n";
	openvdb::io::File* vdbfile = new openvdb::io::File(filename);
	vdbfile->open();

	// Read grid		
	openvdb::GridBase::Ptr baseGrid;
	openvdb::io::File::NameIterator nameIter = vdbfile->beginName();
	std::string name = vdbfile->beginName().gridName();
	for (openvdb::io::File::NameIterator nameIter = vdbfile->beginName(); nameIter != vdbfile->endName(); ++nameIter) {
		std::cout << "   Grid: " << nameIter.gridName() << std::endl;
		if (nameIter.gridName().compare(gvdb.getScene()->mVName) == 0) name = gvdb.getScene()->mVName;
	}
	std::cout << "   Loading Grid: " << name << std::endl;
	baseGrid = vdbfile->readGrid(name);

	PERF_POP();

	// Initialize GVDB config
	Vector3DF voxelsize;
	int gridtype = 0;

	bool isFloat = false;

	std::cout << "   Configuring GVDB.\n";
	if (baseGrid->isType< FloatGrid543 >()) {
		gridtype = 0;
		isFloat = true;
		mOVDB->grid543F = openvdb::gridPtrCast< FloatGrid543 >(baseGrid);
		voxelsize.Set(mOVDB->grid543F->voxelSize().x(), mOVDB->grid543F->voxelSize().y(), mOVDB->grid543F->voxelSize().z());
		gvdb.Configure(5, 5, 5, 4, 3);
	}
	if (baseGrid->isType< Vec3fGrid543 >()) {
		gridtype = 0;
		isFloat = false;
		mOVDB->grid543VF = openvdb::gridPtrCast< Vec3fGrid543 >(baseGrid);
		voxelsize.Set(mOVDB->grid543VF->voxelSize().x(), mOVDB->grid543VF->voxelSize().y(), mOVDB->grid543VF->voxelSize().z());
		gvdb.Configure(5, 5, 5, 4, 3);
	}
	if (baseGrid->isType< FloatGrid34 >()) {
		gridtype = 1;
		isFloat = true;
		mOVDB->grid34F = openvdb::gridPtrCast< FloatGrid34 >(baseGrid);
		voxelsize.Set(mOVDB->grid34F->voxelSize().x(), mOVDB->grid34F->voxelSize().y(), mOVDB->grid34F->voxelSize().z());
		gvdb.Configure(3, 3, 3, 3, 4);
	}
	if (baseGrid->isType< Vec3fGrid34 >()) {
		gridtype = 1;
		isFloat = false;
		mOVDB->grid34VF = openvdb::gridPtrCast< Vec3fGrid34 >(baseGrid);
		voxelsize.Set(mOVDB->grid34VF->voxelSize().x(), mOVDB->grid34VF->voxelSize().y(), mOVDB->grid34VF->voxelSize().z());
		gvdb.Configure(3, 3, 3, 3, 4);
	}
	gvdb.SetVoxelSize(voxelsize.x, voxelsize.y, voxelsize.z);
	gvdb.SetApron(APRON); //GVDB::LoadVBX uses 1

	float pused = gvdb.MeasurePools();
	std::cout << "   Topology Used: " << pused << " MB\n";

	slong leaf;
	int leaf_start = 0;				// starting leaf		gScene.mVLeaf.x;		
	int n, leaf_max, leaf_cnt = 0;
	Vector3DF vclipmin, vclipmax, voffset;
	vclipmin = gvdb.getScene()->mVClipMin;
	vclipmax = gvdb.getScene()->mVClipMax;

	// Determine Volume bounds
	std::cout << "   Compute volume bounds.\n";
	vdbSkip(mOVDB, leaf_start, gridtype, isFloat);
	for (leaf_max = 0; vdbCheck(mOVDB, gridtype, isFloat); ) {
		vdbOrigin(mOVDB, orig, gridtype, isFloat);
		p0.Set(orig.x(), orig.y(), orig.z());
		if (p0.x > vclipmin.x && p0.y > vclipmin.y && p0.z > vclipmin.z && p0.x < vclipmax.x && p0.y < vclipmax.y && p0.z < vclipmax.z) {		// accept condition
			if (leaf_max == 0) {
				gvdb.mVoxMin = p0; gvdb.mVoxMax = p0;
			}
			else {
				if (p0.x < gvdb.mVoxMin.x) gvdb.mVoxMin.x = p0.x;
				if (p0.y < gvdb.mVoxMin.y) gvdb.mVoxMin.y = p0.y;
				if (p0.z < gvdb.mVoxMin.z) gvdb.mVoxMin.z = p0.z;
				if (p0.x > gvdb.mVoxMax.x) gvdb.mVoxMax.x = p0.x;
				if (p0.y > gvdb.mVoxMax.y) gvdb.mVoxMax.y = p0.y;
				if (p0.z > gvdb.mVoxMax.z) gvdb.mVoxMax.z = p0.z;
			}
			leaf_max++;
		}
		vdbNext(mOVDB, gridtype, isFloat);
	}
	voffset = gvdb.mVoxMin * -1;		// offset to positive space (hack)	

	// Activate Space
	PERF_PUSH("Activate");
	n = 0;
	std::cout << "   Activating space.\n" << std::endl;;

	std::vector< Vector3DF >	leaf_pos;
	std::vector< uint64 >		leaf_ptr;

	vdbSkip(mOVDB, leaf_start, gridtype, isFloat);
	for (leaf_max = 0; vdbCheck(mOVDB, gridtype, isFloat); ) {

		// Read leaf position
		vdbOrigin(mOVDB, orig, gridtype, isFloat);
		p0.Set(orig.x(), orig.y(), orig.z());
		p0 += voffset;

		if (p0.x > vclipmin.x && p0.y > vclipmin.y && p0.z > vclipmin.z && p0.x < vclipmax.x && p0.y < vclipmax.y && p0.z < vclipmax.z) {		// accept condition
			// only accept those in clip volume
			bool bnew = false;
			leaf = gvdb.ActivateSpace(gvdb.GetRootNode(), p0, bnew);
			leaf_ptr.push_back(leaf);
			leaf_pos.push_back(p0);
			if (leaf_max == 0) {
				gvdb.mVoxMin = p0; gvdb.mVoxMax = p0;
				std::cout << "   First leaf: " << (leaf_start + n)
					<< "  (" << p0.x << " " << p0.y << " " << p0.z << ")" << std::endl;
			}
			leaf_max++;
		}
		vdbNext(mOVDB, gridtype, isFloat);
		n++;
	}

	// Finish Topology
	gvdb.FinishTopology();

	PERF_POP();		// Activate

	// Resize Atlas
	//verbosef ( "   Create Atlas. Free before: %6.2f MB\n", cudaGetFreeMem() );
	PERF_PUSH("Atlas");
	gvdb.DestroyChannels();
	gvdb.AddChannel(0, T_FLOAT, APRON, F_LINEAR);
	gvdb.UpdateAtlas();
	PERF_POP();
	//verbosef ( "   Create Atlas. Free after:  %6.2f MB, # Leaf: %d\n", cudaGetFreeMem(), leaf_max );

	// Resize temp 3D texture to match leaf res
	int res0 = gvdb.getRes(0);
	Vector3DI vres0(res0, res0, res0);		// leaf resolution
	Vector3DF vrange0 = gvdb.getRange(0);

	nvdb::Volume3D vtemp(gvdb.mScene);
	vtemp.Resize(T_FLOAT, vres0, 0x0, false);
	//Allocator vtemp;
	//vtemp.TextureCreate(0, T_FLOAT, vres0, true, false);

	vclipmin = gvdb.getScene()->mVClipMin;
	vclipmax = gvdb.getScene()->mVClipMax;

	// Read brick data
	PERF_PUSH("Read bricks");

	// Advance to starting leaf		
	vdbSkip(mOVDB, leaf_start, gridtype, isFloat);

	float* src;
	float* src2 = (float*)malloc(res0*res0*res0 * sizeof(float));		// velocity field
	float mValMin, mValMax;
	mValMin = 1.0E35; mValMax = -1.0E35;

	// Fill atlas from leaf data
	int percl = 0, perc = 0;
	std::cout << "   Loading bricks." << std::endl;
	for (leaf_cnt = 0; vdbCheck(mOVDB, gridtype, isFloat); ) {

		// read leaf position
		vdbOrigin(mOVDB, orig, gridtype, isFloat);
		p0.Set(orig.x(), orig.y(), orig.z());
		p0 += voffset;

		if (p0.x > vclipmin.x && p0.y > vclipmin.y && p0.z > vclipmin.z && p0.x < vclipmax.x && p0.y < vclipmax.y && p0.z < vclipmax.z) {		// accept condition			

			// get leaf	
			if (gridtype == 0) {
				if (isFloat) {
					mOVDB->buf3U = (*mOVDB->iter543F).buffer();
					src = mOVDB->buf3U.data();
				}
				else {
					std::cerr << "Unsupported type" << std::endl;
					return false;
				}
			}
			else {
				if (isFloat) {
					mOVDB->buf4U = (*mOVDB->iter34F).buffer();
					src = mOVDB->buf4U.data();
				}
				else {
					std::cerr << "Unsupported type" << std::endl;
					return false;
				}
			}
			// Copy data from CPU into temporary 3D texture
			vtemp.SetDomain(leaf_pos[leaf_cnt], leaf_pos[leaf_cnt] + vrange0);
			vtemp.CommitFromCPU(src);
			assert(cuCtxSynchronize() == CUDA_SUCCESS);
			//vtemp.AtlasCommitFromCPU(0, (uchar*)src);

			// Copy from 3D texture into Atlas brick
			Node* node = gvdb.getNode(leaf_ptr[leaf_cnt]);
			gvdb.mPool->AtlasCopyTexZYX(0, node->mValue, vtemp.getPtr());
			//assert(cuCtxSynchronize() == CUDA_SUCCESS);

			// Progress percent
			leaf_cnt++; perc = int(leaf_cnt * 100 / leaf_max);
			if (perc != percl) { std::cout << perc << "%" << std::flush; percl = perc; }
		}
		vdbNext(mOVDB, gridtype, isFloat);
	}

	PERF_POP();
	if (mValMin != 1.0E35 || mValMax != -1.0E35)
		std::cout << "\n    Value Range: " << mValMin << " " << mValMax << std::endl;

	assert(cuCtxSynchronize() == CUDA_SUCCESS);
	gvdb.UpdateApron();
	assert(cuCtxSynchronize() == CUDA_SUCCESS);

	free(src2);

	// vdbfile->close ();
	// delete vdbfile;
	// delete mOVDB;
	// mOVDB = 0x0; 

	return true;
}

namespace my
{
	
}