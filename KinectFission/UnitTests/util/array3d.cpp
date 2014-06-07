#include <boost/test/auto_unit_test.hpp>

#include <kifi/util/array3d.h>

using namespace kifi;



BOOST_AUTO_TEST_SUITE( array3d )

BOOST_AUTO_TEST_CASE( ctor )
{
	util::array3d< int, 2, 2, 2 > a;

	BOOST_REQUIRE( 8 == a.size() );
}

BOOST_AUTO_TEST_CASE( access )
{
	util::array3d< int, 2, 2, 2 > a;

	a.assign( 7 );

	for( int x = 0; x < 2; x++ )
		for( int y = 0; y < 2; y++ )
			for( int z = 0; z < 2; z++ )
				BOOST_REQUIRE( 7 == a( x, y, z ) );

	a( 0, 0, 0 ) = 2;
	a( 1, 0, 1 ) = 3;

  //BOOST_REQUIRE( 7 == a( 0, 0, 0 ) );
	BOOST_REQUIRE( 7 == a( 0, 0, 1 ) );
	BOOST_REQUIRE( 7 == a( 0, 1, 0 ) );
	BOOST_REQUIRE( 7 == a( 0, 1, 1 ) );

	BOOST_REQUIRE( 7 == a( 1, 0, 0 ) );
  //BOOST_REQUIRE( 7 == a( 1, 0, 1 ) );
	BOOST_REQUIRE( 7 == a( 1, 1, 0 ) );
	BOOST_REQUIRE( 7 == a( 1, 1, 1 ) );

	BOOST_REQUIRE( 2 == a( 0, 0, 0 ) );
	BOOST_REQUIRE( 3 == a( 1, 0, 1 ) );
}

BOOST_AUTO_TEST_SUITE_END()