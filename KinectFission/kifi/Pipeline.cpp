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
	m_eyeToWorld.cols[ 3 ].z() = 2.0f;
}



#pragma warning( push )
#pragma warning( disable : 4100 )

void Pipeline::Integrate
(
	util::vector2d< float > rawDepthMap,
	util::float4x4 const & worldToEye
)
{
#if 0
	m_eyeToWorld = util::invert_transform( worldToEye );
	m_integrator.Integrate( m_volume, rawDepthMap, m_camParams, worldToEye );
#else
	std::printf( "frame %d:\n", m_iFrame );

	if( m_iFrame > 0 )
		// TODO: Swap dst and src
		m_eyeToWorld = m_icp.Align( rawDepthMap, m_eyeToWorld, m_tmpSynthPointCloud, m_camParams );
	else
		m_eyeToWorld = util::invert_transform( worldToEye );

	util::float4x4 tmp = util::invert_transform( m_eyeToWorld );

	// TODO: Change interface of integrator to accept eyeToWorld
	m_integrator.Integrate( m_volume, rawDepthMap, m_camParams, tmp );

	m_mesher.Mesh( m_volume, m_tmpSynthPointCloud );

	auto ref = util::invert_transform( worldToEye );
	double err = 0.0;
	for( int i = 0; i < 4; i++ )
		for( int j = 0; j < 4; j++ )
			err += std::abs( (double) m_eyeToWorld( i, j ) - (double) ref( i, j ) );
	std::printf( "tR err: %f\n", err );
	
	++m_iFrame;
#endif
}

#pragma warning( pop )



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