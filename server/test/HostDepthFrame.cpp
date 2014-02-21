#include <boost/test/auto_unit_test.hpp>

#include <utility>

#include <server/HostDepthFrame.h>



BOOST_AUTO_TEST_SUITE( HostDepthFrame )

BOOST_AUTO_TEST_CASE( default_ctor )
{
	kppl::HostDepthFrame df;

	BOOST_REQUIRE( 0 == df.Width() );
	BOOST_REQUIRE( 0 == df.Height() );
	BOOST_REQUIRE( 0 == df.Resolution() );
}

BOOST_AUTO_TEST_CASE( ctor )
{
	kppl::HostDepthFrame df( 1, 1 );

	BOOST_REQUIRE( 1 == df.Width() );
	BOOST_REQUIRE( 1 == df.Height() );
	BOOST_REQUIRE( 1 == df.Resolution() );
}

BOOST_AUTO_TEST_CASE( Resize )
{
	kppl::HostDepthFrame df;
	df.Resize( 5, 2 );

	BOOST_REQUIRE(  5 == df.Width() );
	BOOST_REQUIRE(  2 == df.Height() );
	BOOST_REQUIRE( 10 == df.Resolution() );
}

BOOST_AUTO_TEST_CASE( access )
{
	kppl::HostDepthFrame df( 1, 1 );
	df( 0, 0 ) = 0.5f;

	BOOST_REQUIRE_CLOSE( 0.5f, df( 0, 0 ), 0.1f );

	df.Resize( 3, 3 );
	df( 2, 2 ) = 1.2f;

	BOOST_REQUIRE_CLOSE( 1.2f, df( 2, 2 ), 0.1f );
}

BOOST_AUTO_TEST_SUITE_END()

