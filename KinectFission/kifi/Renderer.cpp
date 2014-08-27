#include <kifi/Renderer.h>



namespace kifi {

void Renderer::Bin
(
	std::vector< VertexPositionNormal > const & pointCloud,
	util::float4x4 const & worldToClip,

	util::vector2d< VertexPositionNormal > & outPointBuffer
)
{
	using namespace util;

	// TODO: Optimize this entire routine

	m_depthBuffer.resize( outPointBuffer.width(), outPointBuffer.height() );
	std::fill( m_depthBuffer.begin(), m_depthBuffer.end(), std::numeric_limits< float >::max() );

	// float3( 0, 0, 0 ) is interpreted as invalid
	// TODO: Investigate performance of point validity mask (vector< bool >)
	std::fill( outPointBuffer.begin(), outPointBuffer.end(), VertexPositionNormal( float3( 0.0f ), float3( 0.0f ) ) );
	
	float halfWidth = outPointBuffer.width() * 0.5f;
	float halfHeight = outPointBuffer.height() * 0.5f;

	for( std::size_t i = 0; i < pointCloud.size(); ++i )
	{
		float4 point( pointCloud[ i ].position, 1.0f );

		point = homogenize( worldToClip * point );
		int u = (int) (point.x() * halfWidth + halfWidth);
		int v = (int) outPointBuffer.height() - 1 - (int) (point.y() * halfHeight + halfHeight);

		// TODO: Experiment with & and arithmetic zeroing (cmov)
		if( u >= 0 && u < outPointBuffer.width()  &&
			v >= 0 && v < outPointBuffer.height() &&
			point.z() >= -1.0f && point.z() <= 1.0f )
		{
			float depth = m_depthBuffer( u, v );
			if( point.z() < depth )
			{
				m_depthBuffer( u, v ) = point.z();
				outPointBuffer( u, v ) = pointCloud[ i ];
			}
		}
	}
}

} // namespace