#include <kifi/Pipeline.h>

#include <kifi/util/chrono.h>



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
	m_eyeToWorld.col3.z = 2;
}



#pragma warning( push )
#pragma warning( disable : 4100 )

void Pipeline::Integrate
(
	util::vector2d< float > rawDepthMap,
	util::float4x4 const & worldToEye
)
{
	util::chrono::stop_watch sw;

#if 0
	m_eyeToWorld = worldToEye;
	util::invert_transform( m_eyeToWorld );
	sw.restart();
	m_integrator.Integrate( m_volume, rawDepthMap, m_camParams, worldToEye );
	sw.take_time( "tIntegrate" );
#else
	if( m_iFrame > 0 )
		m_eyeToWorld = m_icp.Align( rawDepthMap, m_eyeToWorld, m_tmpSynthPointBuffer, m_eyeToWorld, m_camParams );
	sw.take_time( "tICP" );

	util::float4x4 tmp = m_eyeToWorld;
	util::invert_transform( tmp );

	sw.restart();
	m_integrator.Integrate( m_volume, rawDepthMap, m_camParams, tmp );
	sw.take_time( "tIntegrate" );

	m_mesher.Mesh( m_volume, m_tmpSynthPointCloud );
	
	m_tmpSynthPointBuffer.resize( m_camParams.ResolutionPixels().x, m_camParams.ResolutionPixels().y );
	m_renderer.Bin( m_tmpSynthPointCloud, m_camParams.EyeToClipRH() * tmp, m_tmpSynthPointBuffer );

	++m_iFrame;
#endif

	sw.print_times();
}

#pragma warning( pop )



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

util::float4x4 const & Pipeline::EyeToWorld() const
{
	return m_eyeToWorld;
}

} // namespace