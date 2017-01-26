#include <kifi/Renderer.h>

// HACK
#include <kifi/util/chrono.h>



namespace kifi {

void Renderer::Render
(
	std::vector< util::float3 > const & pointCloud,
	util::float4x4 const & worldToClip,

	util::vector2d< int > & outRgba 
)
{
	std::memset( outRgba.data(), 0, outRgba.size() * 4 );

	float halfWidth = outRgba.width() * 0.5f;
	float halfHeight = outRgba.height() * 0.5f;

	for( std::size_t i = 0; i < pointCloud.size(); ++i )
	{
		util::float4 point( pointCloud[ i ], 1.0f );

		point = util::homogenize( worldToClip * point );
		int u = (int) (point.x * halfWidth + halfWidth);
		int v = (int) outRgba.height() - 1 - (int) (point.y * halfHeight + halfHeight);

		if( u >= 0 && u < outRgba.width() &&
			v >= 0 && v < outRgba.height() )
				outRgba( u, v ) = ~0u;
	}
}

void Renderer::Render
(
	Volume const & volume,
	util::float4x4 const & worldToClip,

	util::vector2d< int > & outRgba 
)
{
	std::memset( outRgba.data(), 0, outRgba.size() * 4 );

	float halfWidth = outRgba.width() * 0.5f;
	float halfHeight = outRgba.height() * 0.5f;

	for( std::size_t i = 0; i < volume.Data().size(); ++i )
	{
		if( volume.Data().values_cbegin()[ i ].Weight() == 0 || std::abs( volume.Data().values_cbegin()[ i ].Distance() ) > volume.VoxelLength() ) continue;

		unsigned x, y, z;
		util::unpack( volume.Data().keys_cbegin()[ i ], x, y, z );
		util::float4 point( volume.VoxelCenter( x, y, z ), 1.0f );

		point = util::homogenize( worldToClip * point );
		int u = (int) (point.x * halfWidth + halfWidth);
		int v = (int) outRgba.height() - 1 - (int) (point.y * halfHeight + halfHeight);

		if( u >= 0 && u < outRgba.width() &&
			v >= 0 && v < outRgba.height() )
				outRgba( u, v ) = ~0u;
	}
}



void Renderer::Bin
(
	std::vector< util::float3 > const & pointCloud,
	util::float4x4 const & worldToClip,

	util::vector2d< util::float3 > & outPointBuffer
)
{
	// TODO: Optimize this entire routine

	m_depthBuffer.resize( outPointBuffer.width(), outPointBuffer.height() );
	std::fill( m_depthBuffer.begin(), m_depthBuffer.end(), std::numeric_limits< float >::max() );

	// float3( 0, 0, 0 ) is interpreted as invalid
	// TODO: Investigate performance of point validity mask (vector< bool >)
	std::fill( outPointBuffer.begin(), outPointBuffer.end(), util::float3( 0.0f ) );

	float halfWidth = outPointBuffer.width() * 0.5f;
	float halfHeight = outPointBuffer.height() * 0.5f;

	for( std::size_t i = 0; i < pointCloud.size(); ++i )
	{
		util::float3 tmp = pointCloud[ i ];
		util::float4 point( tmp, 1.0f );

		point = util::homogenize( worldToClip * point );
		int u = (int) (point.x * halfWidth + halfWidth);
		int v = (int) outPointBuffer.height() - 1 - (int) (point.y * halfHeight + halfHeight);

		if( u >= 0 && u < outPointBuffer.width()  &&
			v >= 0 && v < outPointBuffer.height() &&
			point.z >= -1.0f && point.z <= 1.0f )
		{
			float depth = m_depthBuffer( u, v );
			if( point.z < depth )
			{
				m_depthBuffer( u, v ) = point.z;
				outPointBuffer( u, v ) = tmp;
			}
		}
	}
}

} // namespace