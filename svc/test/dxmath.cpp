#include <boost/test/auto_unit_test.hpp>

#include <reference/dxmath.h>



BOOST_AUTO_TEST_SUITE( dxmath )

BOOST_AUTO_TEST_CASE( homogenize )
{
	svc::float4 v( 1.0f, 1.0f, 1.0f, 2.0f );
	svc::vec _v = svc::load( v );
	_v = svc::homogenize( _v );
	v = svc::store( _v );

	BOOST_REQUIRE( v.x == 0.5f );
	BOOST_REQUIRE( v.w == 1.0f );
}

BOOST_AUTO_TEST_CASE( clamp )
{
	BOOST_REQUIRE( svc::clamp( -3, 0, 5 ) == 0 );
	BOOST_REQUIRE( svc::clamp( 0, 0, 5 ) == 0 );
	BOOST_REQUIRE( svc::clamp( 2, 0, 5 ) == 2 );
	BOOST_REQUIRE( svc::clamp( 5, 0, 5 ) == 5 );
	BOOST_REQUIRE( svc::clamp( 7, 0, 5 ) == 5 );
}

BOOST_AUTO_TEST_CASE( powerOf2 )
{
	BOOST_REQUIRE( svc::powerOf2( 1 ) );
	BOOST_REQUIRE( svc::powerOf2( 2 ) );
	BOOST_REQUIRE( svc::powerOf2( 4 ) );
	BOOST_REQUIRE( svc::powerOf2( 2048 ) );
	BOOST_REQUIRE( svc::powerOf2( 65536 ) );
	BOOST_REQUIRE( svc::powerOf2( 2048 * 2048 ) );

	BOOST_REQUIRE( ! svc::powerOf2( 0 ) );
	BOOST_REQUIRE( ! svc::powerOf2( 3 ) );
	BOOST_REQUIRE( ! svc::powerOf2( 13 ) );
	BOOST_REQUIRE( ! svc::powerOf2( 26 ) );
	BOOST_REQUIRE( ! svc::powerOf2( 48 ) );
}

BOOST_AUTO_TEST_SUITE_END()