#pragma once

#include <vector>

#include <DirectXMath.h>



namespace svc {

class Volume;

class Splatter
{
public:
	static void Splat( Volume const & volume, std::vector< dlh::float4 > & outVertices );
};

}