#include <boost/test/auto_unit_test.hpp>

#include <utility>

#include <server/DepthFrame.h>



BOOST_AUTO_TEST_SUITE( DepthFrame )

BOOST_AUTO_TEST_CASE( default_ctor )
{
	kppl::DepthFrame df;

	BOOST_REQUIRE( 0 == df.Width() );
	BOOST_REQUIRE( 0 == df.Height() );
}

BOOST_AUTO_TEST_CASE( ctor )
{
	kppl::DepthFrame df( 1, 1 );
	df.data()[ 0 ] = 1234;

	BOOST_REQUIRE( 1 == df.Width() );
	BOOST_REQUIRE( 1 == df.Height() );
	BOOST_REQUIRE_CLOSE( 1.234f, df( 0, 0 ), 0.1f );
}

BOOST_AUTO_TEST_SUITE_END()

