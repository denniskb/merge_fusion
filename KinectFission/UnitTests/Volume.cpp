#include <boost/test/auto_unit_test.hpp>

#include <kifi/util/math.h>

#include <kifi/Volume.h>

using namespace kifi;



BOOST_AUTO_TEST_SUITE( VolumeTest )

BOOST_AUTO_TEST_CASE( ctor )
{
	Volume v( 128, 2.0f, 0.02f );

	BOOST_REQUIRE( v.Resolution() == 128 );
	BOOST_REQUIRE( v.SideLength() == 2.0f );
	BOOST_REQUIRE( v.TruncationMargin() == 0.02f );
	BOOST_REQUIRE( v.VoxelLength() == 2.0f / 128.0f );

	BOOST_REQUIRE( v.Minimum().x == -1.0f );
	BOOST_REQUIRE( v.Maximum().y == 1.0f );
}

BOOST_AUTO_TEST_CASE( VoxelCenter_VoxelIndex )
{
	Volume v( 128, 2.0f, 0.02f );
	
	util::vec3 vc = v.VoxelCenter( 33, 21, 92 );
	util::vec3 ci = v.VoxelIndex( vc );

	BOOST_REQUIRE_CLOSE( 33.5f, ci.x, 0.1f );
	BOOST_REQUIRE_CLOSE( 21.5f, ci.y, 0.1f );
	BOOST_REQUIRE_CLOSE( 92.5f, ci.z, 0.1f );
}

BOOST_AUTO_TEST_SUITE_END()