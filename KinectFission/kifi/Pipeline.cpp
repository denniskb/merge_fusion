#include <kifi/Pipeline.h>



namespace kifi {

Pipeline::Pipeline
(
	DepthSensorParams const & cameraParams,

	int volumeResolution, 
	float volumeSideLength, 
	float truncationMargin
) :
	m_camParams( cameraParams ),
	m_volume( volumeResolution, volumeSideLength, truncationMargin )
{
}



void Pipeline::Integrate
(
	util::vector2d< float > rawDepthMap,
	util::float4x4 const & worldToEye
)
{
	m_integrator.Integrate( m_volume, rawDepthMap, m_camParams, worldToEye );
}



Volume const & Pipeline::Volume() const
{
	return m_volume;
}

} // namespace