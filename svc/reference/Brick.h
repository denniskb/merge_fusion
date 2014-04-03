#pragma once

#include "Voxel.h"



namespace svc {

template< int BrickRes >
class Brick
{
public:
	Voxel voxels[ BrickRes * BrickRes * BrickRes ];
};

}