#pragma once

#include <array>

#include "Voxel.h"



namespace svc {

class Brick : public std::array< Voxel, 8 >
{
public:
	static void Index1Dto3D
	( 
		unsigned index, 
		unsigned & outX,
		unsigned & outY, 
		unsigned & outZ
	);
};

}