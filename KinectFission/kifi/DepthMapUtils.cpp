#include <cassert>
#include <cstdint>

#include <kifi/util/vector2d.h>

#include <kifi/DepthMapUtils.h>



namespace kifi {

// static
void DepthMapUtils::MillimetersToMeters
(
	util::vector2d< unsigned short > const & inMillimeters,
	util::vector2d< float > & outMeters
)
{
	assert( 0 == inMillimeters.width() % 4 );

	outMeters.resize( inMillimeters.width(), inMillimeters.height() );

	float * dst = outMeters.data();
	for
	(
		std::uint64_t const * src = reinterpret_cast< std::uint64_t const * >( inMillimeters.data() ),
		* end = reinterpret_cast< std::uint64_t const * >( inMillimeters.data() + inMillimeters.size() );
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
void DepthMapUtils::DepthToPCL
( 
	util::vector2d< float > const & depthMap,
	DepthSensorParams const & cameraParams,

	util::vector2d< util::float4 > & outPCL
)
{
	using namespace util;

	assert( 0 == depthMap.width() % 4 );

	outPCL.resize( depthMap.width(), depthMap.height() );

	vector flInv = set
	(
		1.0f / cameraParams.FocalLengthNorm().x() / cameraParams.ResolutionPixels().x(),
		1.0f / cameraParams.FocalLengthNorm().y() / cameraParams.ResolutionPixels().y(),
		1.0f, 1.0f 
	);

	vector ppOverFl = set
	(
		(0.5f - cameraParams.PrincipalPointNorm().x() * cameraParams.ResolutionPixels().x()),
		(cameraParams.PrincipalPointNorm().y() * cameraParams.ResolutionPixels().y() - 0.5f),
		0.0f, 
		0.0f
	) * flInv;

	vector mask0001 = set( 0.0f, 0.0f, 0.0f, 1.0f );

	for( std::size_t v = 0; v < depthMap.height(); v++ )
	{
		vector point = set( 0.0f, - (float) v, -1.0f, 0.0f );

		for( std::size_t u = 0; u < depthMap.width(); u += 4 )
		{
			vector depths = loadu( & depthMap( u, v ) );

			vector point0 = loadss( (float) (u + 0) ) + point; 
			vector point1 = loadss( (float) (u + 1) ) + point;
			vector point2 = loadss( (float) (u + 2) ) + point;
			vector point3 = loadss( (float) (u + 3) ) + point;

			point0 = point0 * flInv + ppOverFl;
			point1 = point1 * flInv + ppOverFl;
			point2 = point2 * flInv + ppOverFl;
			point3 = point3 * flInv + ppOverFl;
			
			vector depthx = broadcast< 0 >( depths );
			vector depthy = broadcast< 1 >( depths );
			vector depthz = broadcast< 2 >( depths );
			vector depthw = broadcast< 3 >( depths );

			point0 = point0 * depthx + mask0001;
			point1 = point1 * depthy + mask0001;
			point2 = point2 * depthz + mask0001;
			point3 = point3 * depthw + mask0001;

			float * dst = reinterpret_cast< float * >( & outPCL( u, v ) );
			storeu( dst +  0, point0 );
			storeu( dst +  4, point1 );
			storeu( dst +  8, point2 );
			storeu( dst + 12, point3 );
		}
	}
}

} // namespace