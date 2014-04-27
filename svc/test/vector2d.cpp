#include <boost/test/auto_unit_test.hpp>

#include <reference/vector2d.h>



BOOST_AUTO_TEST_SUITE( vector2d )

BOOST_AUTO_TEST_CASE( default_ctor )
{
	svc::vector2d< float > df;

	BOOST_REQUIRE( 0 == df.width() );
	BOOST_REQUIRE( 0 == df.height() );
	BOOST_REQUIRE( 0 == df.size() );
}

BOOST_AUTO_TEST_CASE( ctor )
{
	svc::vector2d< float > df( 1, 1 );

	BOOST_REQUIRE( 1 == df.width() );
	BOOST_REQUIRE( 1 == df.height() );
	BOOST_REQUIRE( 1 == df.size() );
}

BOOST_AUTO_TEST_CASE( resize )
{
	svc::vector2d< float > df;
	df.resize( 5, 2 );

	BOOST_REQUIRE(  5 == df.width() );
	BOOST_REQUIRE(  2 == df.height() );
	BOOST_REQUIRE( 10 == df.size() );
}

BOOST_AUTO_TEST_CASE( access )
{
	svc::vector2d< float > df( 1, 1 );
	df( 0, 0 ) = 0.5f;

	BOOST_REQUIRE_CLOSE( 0.5f, df( 0, 0 ), 0.1f );

	df.resize( 3, 3 );
	df( 2, 2 ) = 1.2f;

	BOOST_REQUIRE_CLOSE( 1.2f, df( 2, 2 ), 0.1f );
}

BOOST_AUTO_TEST_SUITE_END()

