#pragma once

#include <utility>
#include <vector>

#include <kifi/util/math.h>
#include <kifi/util/vector2d.h>

#include <kifi/DepthSensorParams.h>



namespace kifi {

class ICP
{
public:
	util::float4x4 Align
	(
		util::vector2d< float > const & rawDepthMap,
		util::float4x4 const & rawDepthMapWorldToEyeGuess,
		
		util::vector2d< util::float3 > const & synthDepthBuffer,
		util::float4x4 const & synthDepthBufferWorldToEye,

		DepthSensorParams const & cameraParams
	);

private:
	std::vector< std::pair< util::float3, util::float3 > > m_assocs;

	util::float4x4 AlignStep
	(
		util::vector2d< float > const & rawDepthMap,
		util::float4x4 const & rawDepthMapWorldToEyeGuess,
		
		util::vector2d< util::float3 > const & synthDepthBuffer,
		util::float4x4 const & synthDepthBufferWorldToEye,

		DepthSensorParams const & cameraParams
	);
};

} // namespace