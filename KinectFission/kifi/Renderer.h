#pragma once

#include <vector>

#include <kifi/util/math.h>
#include <kifi/util/vector2d.h>

#include <kifi/Mesher.h>



namespace kifi {

class Renderer
{
public:
	void Bin
	(
		std::vector< VertexPositionNormal > const & pointCloud,
		util::float4x4 const & worldToClip,

		util::vector2d< VertexPositionNormal > & outPointBuffer
	);

private:
	util::vector2d< float > m_depthBuffer;
};

} // namespace