#pragma once

#include <utility>
#include <vector>

#include <kifi/util/math.h>
#include <kifi/util/vector2d.h>

#include <kifi/DepthSensorParams.h>
#include <kifi/Mesher.h>



namespace kifi {

class ICP
{
public:
	util::float4x4 Align
	(
		util::vector2d< float > const & rawDepthMap,
		util::float4x4 const & rawEyeToWorldGuess,
		
		std::vector< VertexPositionNormal > const & synthPointCloud,

		DepthSensorParams const & cameraParams
	);

private:
	std::vector< std::pair< util::float3, util::float3 > > m_assocs;

	util::float4x4 AlignStep
	(
		util::vector2d< float > const & rawDepthMap,
		util::float4x4 const & rawEyeToWorldGuess,
		
		std::vector< VertexPositionNormal > const & synthPointCloud,

		DepthSensorParams const & cameraParams
	);
};

} // namespace