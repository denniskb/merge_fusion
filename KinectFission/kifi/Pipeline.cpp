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
	m_eyeToWorld( util::float4x4::identity() )
{
	m_eyeToWorld.col3.z = 2.0f;
}



void Pipeline::Integrate
(
	util::vector2d< float > rawDepthMap,
	util::float4x4 const & worldToEye
)
{
#if 1
	m_eyeToWorld = worldToEye;
	util::invert_transform( m_eyeToWorld );
	m_integrator.Integrate( m_volume, rawDepthMap, m_camParams, worldToEye );
#else
	if( m_iFrame > 0 )
	//	// TODO: Swap dst and src
		m_eyeToWorld = m_icp.Align( rawDepthMap, m_eyeToWorld, m_tmpSynthPointBuffer, m_eyeToWorld, m_camParams );

	util::float4x4 tmp = m_eyeToWorld;
	util::invert_transform( tmp );

	// TODO: Change interface of integrator to accept eyeToWorld
	//if( m_iFrame == 0 )
		m_integrator.Integrate( m_volume, rawDepthMap, m_camParams, tmp );

	m_mesher.Mesh( m_volume, m_tmpSynthPointCloud );
	//
	//m_tmpSynthPointBuffer.resize( m_camParams.ResolutionPixels().x, m_camParams.ResolutionPixels().y );
	m_tmpSynthPointBuffer.resize( 1024, 768 );
	m_renderer.Bin( m_tmpSynthPointCloud, m_camParams.EyeToClipRH() * tmp, m_tmpSynthPointBuffer );

	++m_iFrame;
#endif
}



void Pipeline::Mesh( std::vector< VertexPositionNormal > & outVertices )
{
	m_mesher.Mesh( m_volume, outVertices );
}

void Pipeline::Mesh( std::vector< VertexPositionNormal > & outVertices, std::vector< unsigned > & outIndices )
{
	m_mesher.Mesh( m_volume, outVertices, outIndices );
}



Volume const & Pipeline::Volume() const
{
	return m_volume;
}

util::float4x4 const & Pipeline::EyeToWorld() const
{
	return m_eyeToWorld;
}

} // namespace