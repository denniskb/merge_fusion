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

} // namespace