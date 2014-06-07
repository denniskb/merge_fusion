#include <boost/test/auto_unit_test.hpp>

#include <kifi/util/vector2d.h>

using namespace kifi;



BOOST_AUTO_TEST_SUITE( util_test )
BOOST_AUTO_TEST_SUITE( vector2d_test )

BOOST_AUTO_TEST_CASE( default_ctor )
{
	util::vector2d< int > df;

	BOOST_REQUIRE( 0 == df.width() );
	BOOST_REQUIRE( 0 == df.height() );
	BOOST_REQUIRE( 0 == df.size() );
}

BOOST_AUTO_TEST_CASE( ctor )
{
	util::vector2d< int > df( 1, 1 );

	BOOST_REQUIRE( 1 == df.width() );
	BOOST_REQUIRE( 1 == df.height() );
	BOOST_REQUIRE( 1 == df.size() );
}

BOOST_AUTO_TEST_CASE( resize )
{
	util::vector2d< int > df;
	df.resize( 5, 2 );

	BOOST_REQUIRE(  5 == df.width() );
	BOOST_REQUIRE(  2 == df.height() );
	BOOST_REQUIRE( 10 == df.size() );
}

BOOST_AUTO_TEST_CASE( access )
{
	util::vector2d< int > df( 1, 1 );
	df( 0, 0 ) = 5;

	BOOST_REQUIRE( 5 == df( 0, 0 ) );

	df.resize( 3, 3 );
	df( 2, 2 ) = 12;

	BOOST_REQUIRE( 12 == df( 2, 2 ) );
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()