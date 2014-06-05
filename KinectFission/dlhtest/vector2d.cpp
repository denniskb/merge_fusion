#include <boost/test/auto_unit_test.hpp>

#include <dlh/vector2d.h>



BOOST_AUTO_TEST_SUITE( vector2d )

BOOST_AUTO_TEST_CASE( default_ctor )
{
	dlh::vector2d< int > df;

	BOOST_REQUIRE( 0 == df.width() );
	BOOST_REQUIRE( 0 == df.height() );
	BOOST_REQUIRE( 0 == df.size() );
}

BOOST_AUTO_TEST_CASE( ctor )
{
	dlh::vector2d< int > df( 1, 1 );

	BOOST_REQUIRE( 1 == df.width() );
	BOOST_REQUIRE( 1 == df.height() );
	BOOST_REQUIRE( 1 == df.size() );
}

BOOST_AUTO_TEST_CASE( resize )
{
	dlh::vector2d< int > df;
	df.resize( 5, 2 );

	BOOST_REQUIRE(  5 == df.width() );
	BOOST_REQUIRE(  2 == df.height() );
	BOOST_REQUIRE( 10 == df.size() );
}

BOOST_AUTO_TEST_CASE( access )
{
	dlh::vector2d< int > df( 1, 1 );
	df( 0, 0 ) = 5;

	BOOST_REQUIRE( 5 == df( 0, 0 ) );

	df.resize( 3, 3 );
	df( 2, 2 ) = 12;

	BOOST_REQUIRE( 12 == df( 2, 2 ) );
}

BOOST_AUTO_TEST_SUITE_END()