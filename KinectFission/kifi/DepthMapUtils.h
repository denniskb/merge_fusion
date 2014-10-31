#pragma once

#include <kifi/util/math.h>
#include <kifi/util/vector2d.h>

#include <kifi/DepthSensorParams.h>



namespace kifi {

class DepthMapUtils
{
public:
	static void Millimeters2Meters
	(
		util::vector2d< unsigned short > const & depthMillimeters,
		util::vector2d< float > & depthMeters
	);

	static void BilateralFilter
	(
		util::vector2d< float > const & rawDepthMeters,
		int kernelRadius, float smoothnessFactor,
		DepthSensorParams const & params,

		util::vector2d< float > & smoothDepthMeters
	);

	static void Depth2Normals
	(
		util::vector2d< float > const & depthMeters,
		DepthSensorParams const & params,
	
		util::vector2d< util::float3 > & normals
	);
};

} // namespace