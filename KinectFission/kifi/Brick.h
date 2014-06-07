#pragma once

#include <array>

#include <kifi/Voxel.h>



namespace kifi {

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

} // namespace