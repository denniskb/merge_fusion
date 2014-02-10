#include <boost/test/auto_unit_test.hpp>

#include <server/Voxel.h>

BOOST_AUTO_TEST_SUITE( Voxel )

BOOST_AUTO_TEST_CASE( Update )
{
	float const tm = 0.271f;

	kppl::Voxel v;
	BOOST_REQUIRE_CLOSE( v.Distance( tm ) + 1.0f, 1.0f, 2.0f );
	BOOST_REQUIRE( 0 == v.Weight() );

	v.Update( 0.1, tm );
	BOOST_REQUIRE_CLOSE( v.Distance( tm ), 0.1f, 2.0f );
	BOOST_REQUIRE( 1 == v.Weight() );

	kppl::Voxel v2;
	v2.Update( 0.5f, tm );
	BOOST_REQUIRE_CLOSE( v2.Distance( tm ), tm, 2.0f );
}

BOOST_AUTO_TEST_SUITE_END()