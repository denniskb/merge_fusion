#include <boost/test/auto_unit_test.hpp>

#include <cuda/DepthFrame.h>

#include <dlh/vector2d.h>



BOOST_AUTO_TEST_SUITE( DepthFrame )

BOOST_AUTO_TEST_CASE( load_and_store )
{
	dlh::vector2d< float > hf( 4, 3 );
	svcu::DepthFrame df;

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