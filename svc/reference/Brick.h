#pragma once

#include "Voxel.h"



namespace svc {

class Brick
{
public:
	static int const RESOLUTION = 2;
	static int const SLICE = RESOLUTION * RESOLUTION;
	static int const VOLUME = SLICE * RESOLUTION;

	Voxel voxels[ VOLUME ];
};

}