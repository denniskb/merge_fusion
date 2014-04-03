#include <boost/test/auto_unit_test.hpp>

#include <reference/Volume.h>



BOOST_AUTO_TEST_SUITE( Volume )

BOOST_AUTO_TEST_CASE( ctor )
{
	svc::Volume< 2 > v( 128, 2.0f, 0.02f );

	BOOST_REQUIRE( v.Resolution() == 128 );
	BOOST_REQUIRE( v.SideLength() == 2.0f );
	BOOST_REQUIRE( v.VoxelLength() == 2.0f / 128.0f );
	BOOST_REQUIRE( v.TruncationMargin() == 0.02f );

	BOOST_REQUIRE( v.BrickSlice() == 4 );
	BOOST_REQUIRE( v.BrickVolume() == 8 );
	BOOST_REQUIRE( v.NumBricksInVolume() == 64 );

	BOOST_REQUIRE( v.Minimum().x == -1.0f );
	BOOST_REQUIRE( v.Maximum().y == 1.0f );
}

BOOST_AUTO_TEST_SUITE_END()