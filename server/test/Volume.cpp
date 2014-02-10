#include <boost/test/auto_unit_test.hpp>

#include <server/Volume.h>
#include <server/Voxel.h>



BOOST_AUTO_TEST_SUITE( Volume )

BOOST_AUTO_TEST_CASE( getset )
{
	kppl::Volume vol( 10, 1.0f );
	BOOST_REQUIRE( 10 == vol.Resolution() );

	kppl::Voxel v;
	BOOST_REQUIRE( vol( 0, 0, 0 ) == v );
	BOOST_REQUIRE( vol( 3, 1, 9 ) == v );

	v.Update( 1.0f, 1.0f );
	BOOST_REQUIRE( vol( 2, 7, 1 ) != v );

	vol( 1, 1, 1 ) = v;
	BOOST_REQUIRE( vol( 1, 1, 1 ) == v );

	v.Update( 0.0f, 1.0f );
	vol( 1, 1, 1 ).Update( 0.0f, 1.0f );
	BOOST_REQUIRE( vol( 1, 1, 1 ) == v );
}

BOOST_AUTO_TEST_SUITE_END()