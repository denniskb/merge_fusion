#include <boost/test/auto_unit_test.hpp>

#include <kifi/Voxel.h>

using namespace kifi;



BOOST_AUTO_TEST_SUITE( VoxelTest )

BOOST_AUTO_TEST_CASE( Update )
{
	Voxel v;
	BOOST_REQUIRE( 0 == v.Weight() );

	v.Update( 0.1f );
	BOOST_REQUIRE_CLOSE( v.Distance(), 0.1f, 0.1f );
	BOOST_REQUIRE( 1 == v.Weight() );
}

BOOST_AUTO_TEST_CASE( UpdateWeight )
{
	Voxel v;
	v.Update( 0.005f );

	BOOST_REQUIRE( 1 == v.Weight() );
	BOOST_REQUIRE_CLOSE( v.Distance(), 0.005f, 0.1f );
}

BOOST_AUTO_TEST_CASE( UpdateWeight2 )
{
	Voxel v1;
	v1.Update( 1.0f );
	v1.Update( 0.333f );
	v1.Update( 0.333f );
	v1.Update( 0.333f );

	BOOST_REQUIRE( 4 == v1.Weight() );
	BOOST_REQUIRE_CLOSE( v1.Distance(), 0.5f, 0.1f );
}

BOOST_AUTO_TEST_SUITE_END()