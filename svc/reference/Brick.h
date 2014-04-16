#pragma once

#include <array>

#include "Voxel.h"



namespace svc {

class Brick : public std::array< Voxel, 8 >
{
public:
	static int const RESOLUTION = 2;
	static int const SLICE = RESOLUTION * RESOLUTION;
	static int const VOLUME = SLICE * RESOLUTION;

	static void Index1Dto3D( unsigned index, unsigned & outX, unsigned & outY, unsigned & outZ );
};

}