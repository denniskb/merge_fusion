#include <kifi/Pipeline.h>



namespace kifi {

Pipeline::Pipeline
(
	DepthSensorParams const & cameraParams,

	int volumeResolution, 
	float volumeSideLength, 
	float truncationMargin
) :
	m_volume( volumeResolution, volumeSideLength, truncationMargin ),

	m_camParams( cameraParams ),
	m_eyeToWorld( util::float4x4::identity() )
{
	// HACK
	 m_eyeToWorld.cols[ 3 ].z() += 2.0f;
}



void Pipeline::Integrate( util::vector2d< float > const & rawDepthMap, std::size_t nPoints )
{
	if( ! m_tmpSynthPointCloud.empty() )
		m_eyeToWorld = m_icp.Align( rawDepthMap, m_eyeToWorld, m_tmpSynthPointCloud, m_camParams, nPoints );

	Integrate( rawDepthMap, m_eyeToWorld );
	Mesh( m_tmpSynthPointCloud );
}

void Pipeline::Integrate( util::vector2d< float > const & rawDepthMap, util::float4x4 const & eyeToWorld )
{
	m_eyeToWorld = eyeToWorld;
	m_integrator.Integrate( m_volume, rawDepthMap, m_camParams, eyeToWorld );	
}



void Pipeline::Mesh( std::vector< VertexPositionNormal > & outVertices )
{
	m_mesher.Mesh( m_volume, outVertices );
}

void Pipeline::Mesh( std::vector< VertexPositionNormal > & outVertices, std::vector< unsigned > & outIndices )
{
	m_mesher.Mesh( m_volume, outVertices, outIndices );
}



util::float4x4 const & Pipeline::EyeToWorld() const
{
	return m_eyeToWorld;
}

std::vector< VertexPositionNormal > const & Pipeline::SynthPointCloud()
{
	return m_tmpSynthPointCloud;
}

} // namespace