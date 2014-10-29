#pragma once

#include <kifi/util/math.h>
#include <kifi/util/vector2d.h>

#include <kifi/DepthSensorParams.h>



namespace kifi {

class DepthMapUtils
{
public:
	static void MillimetersToMeters
	(
		util::vector2d< unsigned short > const & inMillimeters,
		util::vector2d< float > & outMeters
	);
};

} // namespace