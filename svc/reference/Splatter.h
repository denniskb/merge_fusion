#pragma once

#include <vector>

#include <flink/math.h>



namespace svc {

class Volume;

class Splatter
{
public:
	static void Splat( Volume const & volume, std::vector< flink::float4 > & outVertices );
};

}