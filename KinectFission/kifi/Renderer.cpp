#include <kifi/Renderer.h>

// HACK
#include <kifi/util/chrono.h>



struct g
{
	kifi::util::float2 d;
	float r;
	kifi::util::float3 n;

	g() :
		r( 0.0f )
	{}
};

//static kifi::util::float2 bi( kifi::util::vector2d< kifi::util::float2 > const & img, float x, float y )
//{
//
//}
/*
static kifi::util::vector2d< g > down_sample( kifi::util::vector2d< g > map )
{
	using namespace kifi::util;

	vector2d< g > result( map.width() / 2, map.height() / 2 );

	for( int y = 0; y < map.height(); y += 2 )
		for( int x = 0; x < map.width(); x += 2 )
		{
			g a, b, c, d;
			a = map( x, y );
			b = map( x + 1, y );
			c = map( x, y + 1 );
			d = map( x + 1, y + 1 );

			float ivstart = std::numeric_limits< float >::max();
			float ivend = -std::numeric_limits< float >::max();
			if( a.d.x < ivstart ) { ivstart = a.d.x; ivend = a.d.y; }
			if( b.d.x < ivstart ) { ivstart = b.d.x; ivend = b.d.y; }
			if( c.d.x < ivstart ) { ivstart = c.d.x; ivend = c.d.y; }
			if( d.d.x < ivstart ) { ivstart = d.d.x; ivend = d.d.y; }
			
			g avg;
			avg.d = float2( std::numeric_limits< float >::max(), -std::numeric_limits< float >::max() );
			int n = 0;
			if( a.r > 0.0f && a.d.x <= ivend ) { avg.d.x = std::min( avg.d.x, a.d.x ); avg.d.y = std::max( avg.d.y, a.d.y ); avg.r += a.r; avg.n.x += a.n.x; avg.n.y += a.n.y; avg.n.z += a.n.z; n++; }
			if( b.r > 0.0f && b.d.x <= ivend ) { avg.d.x = std::min( avg.d.x, b.d.x ); avg.d.y = std::max( avg.d.y, b.d.y ); avg.r += b.r; avg.n.x += b.n.x; avg.n.y += b.n.y; avg.n.z += b.n.z; n++; }
			if( c.r > 0.0f && c.d.x <= ivend ) { avg.d.x = std::min( avg.d.x, c.d.x ); avg.d.y = std::max( avg.d.y, c.d.y ); avg.r += c.r; avg.n.x += c.n.x; avg.n.y += c.n.y; avg.n.z += c.n.z; n++; }
			if( d.r > 0.0f && d.d.x <= ivend ) { avg.d.x = std::min( avg.d.x, d.d.x ); avg.d.y = std::max( avg.d.y, d.d.y ); avg.r += d.r; avg.n.x += d.n.x; avg.n.y += d.n.y; avg.n.z += d.n.z; n++; }

			if( n > 0 ) { avg.r /= n; avg.n.x /= n; avg.n.y /= n; avg.n.z /= n; }

			result( x / 2, y / 2 ) = avg;
		}

	return result;
}

static void fill_holes( kifi::util::vector2d< g > & fine, kifi::util::vector2d< g > const & coarse )
{
	for( int y = 0; y < fine.height(); y++ )
		for( int x = 0; x < fine.width(); x++ )
			//if( fine( x, y ).r == 0.0f )
				fine( x, y ) = coarse( x / 2, y / 2 );
}*/

static kifi::util::vector2d< g > fill_holes2( kifi::util::vector2d< g > const & splat )
{
	using namespace kifi::util;

	vector2d< g > result( splat.width(), splat.height() );

	for( int y = 0; y < result.height(); y++ )
		for( int x = 0; x < result.width(); x++ )
		{
			int kernel = 2;
			
			int sx = x - kernel; int ex = x + kernel;
			int sy = y - kernel; int ey = y + kernel;

			sx = std::max( 0, std::min( (int) result.width() - 1, sx ) );
			sy = std::max( 0, std::min( (int) result.height() - 1, sy ) );

			ex = std::max( 0, std::min( (int) result.width() - 1, ex ) );
			ey = std::max( 0, std::min( (int) result.height() - 1, ey ) );

			float3 n;
			float count = 0;
			int k = 0;
			float depths[ 25 ];
			float mind = 0.0f;
			for( int j = sy; j <= ey; j++ )
				for( int i = sx; i <= ex; i++ )
					if( splat( i, j ).r > 0.0f )
						depths[ k++ ] = splat( i, j ).d.x;

			std::sort( depths, depths + k );

			if( k == 1 )
				mind = depths[ 0 ];
			else if( k > 1 )
			{
				for( int i = 0; i < k/2; i++ )
					mind += depths[ i ];

				mind /= k/2;
			}

			for( int j = sy; j <= ey; j++ )
				for( int i = sx; i <= ex; i++ )
					if( splat( i, j ).r > 0.0f &&
						//abs( i - x ) < splat( i, j ).r &&
						//abs( j - y ) < splat( i, j ).r &&
						abs( splat( i, j ).d.x - mind ) < 2.0f / 512 )
					{
						float distexp = (float) ( ( i - x ) * ( i - x ) + ( j - y ) * ( j - y ) ) + 1.0f;
						float weight = 1.0f / distexp;

						n.x += weight * splat( i, j ).n.x;
						n.y += weight * splat( i, j ).n.y;
						n.z += weight * splat( i, j ).n.z;

						count += weight;
					}

			if( count > 0 )
			{
				n.x /= count;
				n.y /= count;
				n.z /= count;
			}

			result( x, y ).n = n;
		}

	return result;
}



namespace kifi {

void Renderer::Render
(
	std::vector< VertexPositionNormal > const & pointCloud,
	util::float4x4 const & worldToClip,

	util::vector2d< int > & outRgba 
)
{
	//std::memset( outRgba.data(), 0, outRgba.size() * sizeof( int ) );
	
	//m_depthBuffer.resize( outRgba.width(), outRgba.height() );
	//std::fill( m_depthBuffer.begin(), m_depthBuffer.end(), std::numeric_limits< float >::max() );
	static util::vector2d< g > depthRadius( outRgba.width(), outRgba.height() );
	g neutral; neutral.d = util::float2( std::numeric_limits< float >::max(), -std::numeric_limits< float >::max() );
	std::fill( depthRadius.begin(), depthRadius.end(), neutral );

	//static util::vector2d< util::float3 > pointBuffer( outRgba.width(), outRgba.height() );
	//std::fill( pointBuffer.begin(), pointBuffer.end(), util::float3() );

	float halfWidth = outRgba.width() * 0.5f;
	float halfHeight = outRgba.height() * 0.5f;

	for( std::size_t i = 0; i < pointCloud.size(); ++i )
	{
		util::float4 point( pointCloud[ i ].position, 1.0f );

		point = worldToClip * point;
		float w = point.w;
		// r == half voxel len
		float r = (1.0f / 512) * (2.0f * 571.26f / 640) / w * halfWidth;

		point /= w;
		int u = (int) (point.x * halfWidth + halfWidth);
		int v = (int) outRgba.height() - 1 - (int) (point.y * halfHeight + halfHeight);

		if( u >= 0 && u < outRgba.width() &&
			v >= 0 && v < outRgba.height() &&
			point.z >= -1.0f && point.z < 1.0f )
		{
			float depth = depthRadius( u, v ).d.x;
			if( w < depth )
			{
				depthRadius( u, v ).d.x = w;
				depthRadius( u, v ).d.y = w + 1.0f / 512;
				depthRadius( u, v ).r = r;
				depthRadius( u, v ).n = pointCloud[ i ].normal;
			}
		}
	}

	auto test = fill_holes2( depthRadius );

	// TODO: Average and fill gaps, start with simplest solution and work your way up
	for( int i = 0; i < outRgba.size(); i++ )
	{
		int r = (int) (test[ i ].n.x * 255);
		int g = (int) (test[ i ].n.y * 255);
		int b = (int) (test[ i ].n.z * 255);
		outRgba[ i ] = b << 16 | g << 8 | r;
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