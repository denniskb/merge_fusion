#include <kifi/ICP.h>



namespace kifi {

util::float4x4 ICP::AlignStep
(
	util::vector2d< float > const & rawDepthMap,
	util::float4x4 const & rawDepthMapWorldToEyeGuess,
		
	util::vector2d< util::float3 > const & synthDepthBuffer,
	util::float4x4 const & synthDepthBufferWorldToEye,

	DepthSensorParams const & cameraParams
)
{

}

} // namespace