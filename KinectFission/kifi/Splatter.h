#pragma once

#include <vector>

#include <kifi/util/math.h>



namespace kifi {

class Volume;

class Splatter
{
public:
	static void Splat( Volume const & volume, std::vector< util::float4 > & outVertices );
};

} // namespace