#pragma once

#include <vector>

#include "dxmath.h"



namespace svc {

class Volume;

class Splatter
{
public:
	static void Splat( Volume const & volume, std::vector< float4 > & outVertices );
};

}