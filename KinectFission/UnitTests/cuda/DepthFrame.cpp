#include <boost/test/auto_unit_test.hpp>

#include <kifi/cuda/DepthFrame.h>

#include <kifi/util/vector2d.h>

using namespace kifi;



BOOST_AUTO_TEST_SUITE( cuda_test )
BOOST_AUTO_TEST_SUITE( DepthFrameTest )

BOOST_AUTO_TEST_CASE( load_and_store )
{
	util::vector2d< float > hf( 4, 3 );
	cuda::DepthFrame df;

	hf( 0, 0 ) = 7.0f;
	hf( 2, 1 ) = 3.0f;
	df.CopyFrom( hf );

	BOOST_REQUIRE( df.Width() == 4 );
	BOOST_REQUIRE( df.Height() == 3 );

	std::fill( & hf( 0, 0 ), & hf( 3, 2 ), 0.0f );
	df.CopyTo( hf );

	BOOST_REQUIRE( hf.width() == 4 );
	BOOST_REQUIRE( hf.height() == 3 );
	BOOST_REQUIRE( hf( 0, 0 ) == 7.0f );
	BOOST_REQUIRE( hf( 2, 1 ) == 3.0f );
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()