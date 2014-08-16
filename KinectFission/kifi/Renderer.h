#pragma once

#include <vector>

#include <kifi/util/math.h>
#include <kifi/util/vector2d.h>

#include <kifi/Mesher.h>



namespace kifi {

class Renderer
{
public:
	void Render
	(
		std::vector< VertexPositionNormal > const & pointCloud,
		util::float4x4 const & worldToClip,

		util::vector2d< int > & outRgba 
	);

	void Bin
	(
		std::vector< util::float3 > const & pointCloud,
		util::float4x4 const & worldToClip,

		util::vector2d< util::float3 > & outPointBuffer
	);

private:
	util::vector2d< float > m_depthBuffer;
};

} // namespace