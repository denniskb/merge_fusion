#include <algorithm>
#include <cassert>
#include <cstdint>

#include <kifi/util/math.h>
#include <kifi/util/vector2d.h>

#include <kifi/DepthMapUtils.h>



namespace kifi {

// static
void DepthMapUtils::Millimeters2Meters
(
	util::vector2d< unsigned short > const & depthMillimeters,
	util::vector2d< float > & depthMeters
)
{
	assert( 0 == depthMillimeters.width() % 4 );

	depthMeters.resize( depthMillimeters.width(), depthMillimeters.height() );

	float * dst = depthMeters.data();
	for
	(
		std::uint64_t const * src = reinterpret_cast< std::uint64_t const * >( depthMillimeters.data() ),
		* end = reinterpret_cast< std::uint64_t const * >( depthMillimeters.data() + depthMillimeters.size() );
		src < end;
		src++
	)
	{
		std::uint64_t const depths = * src;

		// little endian
		* dst++ = ( depths        & 0xffff) * 0.001f;
		* dst++ = ((depths >> 16) & 0xffff) * 0.001f;
		* dst++ = ((depths >> 32) & 0xffff) * 0.001f;
		* dst++ = ( depths >> 48          ) * 0.001f;
	}
}

// static 
void DepthMapUtils::BilateralFilter
(
	util::vector2d< float > const & rawDepthMeters,
	int kernelRadius, float smoothnessFactor,
	DepthSensorParams const & params,

	util::vector2d< float > & smoothDepthMeters
)
{
	assert( kernelRadius >= 0 );

	if( 0 == kernelRadius )
	{
		smoothDepthMeters = rawDepthMeters;
		return;
	}

	smoothDepthMeters.resize( rawDepthMeters.width(), rawDepthMeters.height() );

	float const pxSizeFactor = 1.0f / (params.FocalLengthNorm().y() * params.ResolutionPixels().y());

	for( int y = 0; y < rawDepthMeters.height(); y++ )
		for( int x = 0; x < rawDepthMeters.width(); x++ )
		{
			float self = rawDepthMeters( x, y );

			if( 0.0f == self )
			{
				smoothDepthMeters( x, y ) = 0.0f;
				continue;
			}

			int xmin = std::max< int >( 0, x - kernelRadius );
			int xmax = std::min< int >( (int) rawDepthMeters.width() - 1, x + kernelRadius );

			int ymin = std::max< int >( 0, y - kernelRadius );
			int ymax = std::min< int >( (int) rawDepthMeters.height() - 1, y + kernelRadius );

			float pxSizeMeters = self * pxSizeFactor;
			pxSizeMeters *= pxSizeMeters;

			float intensity  = 0.0f;
			float sumWeights = 0.0f;
			for( int j = ymin; j <= ymax; j++ )
				for( int i = xmin; i <= xmax; i++ )
				{
					float other = rawDepthMeters( i, j );

					int   dx = i - x;
					int   dy = j - y;
					float dz = self - other;

					float weight = 1.0f / util::exp256( std::sqrtf( (dx*dx + dy*dy) * pxSizeMeters + dz*dz ) * smoothnessFactor );

					intensity  += weight * other;
					sumWeights += weight * (other != 0.0f);
				}
			
			sumWeights += (0.0f == sumWeights);
			intensity /= sumWeights;

			smoothDepthMeters( x, y ) = intensity;
		}
}

// static 
void Depth2Normals
(
	util::vector2d< float > const & depthMeters,
	DepthSensorParams const & params,

	util::vector2d< util::float3 > & normals
)
{
	using namespace util;

	normals.resize( depthMeters.width(), depthMeters.height() );

	float2 flInv = 1.0f / (params.FocalLengthNorm() * (float2) params.ResolutionPixels());

	float2 ppOverFl = float2
	(
		0.5f - params.PrincipalPointNorm().x() * params.ResolutionPixels().x(),
		params.PrincipalPointNorm().y() * params.ResolutionPixels().y() - 0.5f
	) * 
	flInv;

	flInv.y() = -flInv.y();

	for( int y = 0, yend = (int) depthMeters.height() - 1; y < yend; y++ )
		for( int x = 0, xend = (int) depthMeters.width() - 1; x < xend; x++ )
		{
			float self  = depthMeters( x, y );
			float right = depthMeters( x + 1, y );
			float down  = depthMeters( x, y + 1 );

			if( 0.0f == self || 0.0f == right || 0.0f == down )
			{
				normals( x, y ) = float3( 0.0f );
				continue;
			}
			
			float3 pself  = clipNorm2Eye( float2( (float) (x    ), (float) (y    ) ), flInv, ppOverFl, self  );
			float3 pright = clipNorm2Eye( float2( (float) (x + 1), (float) (y    ) ), flInv, ppOverFl, right );
			float3 pdown  = clipNorm2Eye( float2( (float) (x    ), (float) (y + 1) ), flInv, ppOverFl, down  );

			normals( x, y ) = normalize( cross( pdown - pself, pright - pself ) );
		}
}

} // namespace