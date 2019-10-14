#pragma once

#include <gvdb/gvdb.h>

class Vdb2Vbx
{
private:
	Vdb2Vbx() {}

public:
	static bool LoadVDB(nvdb::VolumeGVDB& gvdb, const std::string& filename);
};

