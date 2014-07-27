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
	m_volume( volumeResolution, volumeSideLength, truncationMargin ),

	m_iFrame( 0 ),
	m_worldToEye( util::identity )
{
}



void Pipeline::Integrate
(
	util::vector2d< float > rawDepthMap,
	util::float4x4 const & worldToEye
)
{
#if 1
	m_integrator.Integrate( m_volume, rawDepthMap, m_camParams, worldToEye );
#else
	// TODO: For testing purposes set m_worldToEye to the first worldToView matrix from the depth stream
	// => consequtive matrices should be equal

	if( m_iFrame > 0 )
		m_worldToEye = m_icp.Align( rawDepthMap, m_worldToEye, m_tmpSynthPointBuffer, m_worldToEye, m_camParams );

	m_integrator.Integrate( m_volume, rawDepthMap, m_camParams, m_worldToEye );
	m_mesher.Mesh( m_volume, m_tmpSynthPointCloud );
	m_renderer.Bin( m_tmpSynthPointCloud, m_camParams.EyeToClipRH() * m_worldToEye, m_tmpSynthPointBuffer );
	// Depth map is not a by-product of the rendering step =S

	++m_iFrame;
#endif
}



void Pipeline::Mesh( std::vector< util::float3 > & outVertices )
{
	m_mesher.Mesh( m_volume, outVertices );
}

void Pipeline::Mesh( std::vector< util::float3 > & outVertices, std::vector< unsigned > & outIndices )
{
	m_mesher.Mesh( m_volume, outVertices, outIndices );
}



Volume const & Pipeline::Volume() const
{
	return m_volume;
}

} // namespace