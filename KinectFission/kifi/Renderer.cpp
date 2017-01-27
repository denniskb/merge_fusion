#include <kifi/Renderer.h>

// HACK
#include <kifi/util/chrono.h>



static int pack( int red, int green, int blue )
{
	return red | green << 8 | blue << 16;
}

template< typename T >
static T clamp( T x, T lo, T hi )
{
	return std::max( lo, std::min( hi, x ) );
}

namespace kifi {

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

		if( u < 0 || u >= outRgba.width() ||
			v < 0 || v >= outRgba.height() ||
			point.z < 0.0f || point.z > 1.0f )
			continue;

		float red   = clamp((point.z - 0.5f) * 6.0f, 0.0f, 1.0f);
		float green = std::min(1.0f, point.z * 3.0f) - std::max(0.0f, (point.z - 0.666f) * 3.0f);
		float blue  = 1.0f - clamp((point.z - 0.333f) * 6.0f, 0.0f, 1.0f);

		outRgba( u, v ) = pack
		(
			(int) (255 * red),
			(int) (255 * green),
			(int) (255 * blue)
		);
	}
}



void Renderer::Bin
(
	std::vector< util::float3 > const & pointCloud,
	util::float4x4 const & worldToClip,

	util::vector2d< util::float3 > & outPointBuffer
)
{
	m_depthBuffer.resize( outPointBuffer.width(), outPointBuffer.height() );
	std::fill( m_depthBuffer.begin(), m_depthBuffer.end(), std::numeric_limits< float >::max() );

	std::fill( outPointBuffer.begin(), outPointBuffer.end(), util::float3( 0.0f ) );

	float halfWidth = outPointBuffer.width() * 0.5f;
	float halfHeight = outPointBuffer.height() * 0.5f;

	for( std::size_t i = 0; i < pointCloud.size(); ++i )
	{
		util::float4 point( pointCloud[ i ], 1.0f );

		point = util::homogenize( worldToClip * point );
		int u = (int) (point.x * halfWidth + halfWidth);
		int v = (int) outPointBuffer.height() - 1 - (int) (point.y * halfHeight + halfHeight);

		if( u < 0 || u >= outPointBuffer.width() ||
			v < 0 || v >= outPointBuffer.height() ||
			point.z < 0.0f || point.z > 1.0f )
			continue;

		if( point.z >= m_depthBuffer( u, v ) ) continue;

		m_depthBuffer ( u, v ) = point.z;
		outPointBuffer( u, v ) = pointCloud[ i ];
	}
}

} // namespace