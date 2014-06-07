#include <boost/test/auto_unit_test.hpp>

#include <kifi/util/DirectXMathExt.h>

using namespace kifi;



BOOST_AUTO_TEST_SUITE( DirectXMathExt )

BOOST_AUTO_TEST_CASE( load_store )
{
	util::float4 v( 1, 2, 3, 4 );
	util::vec _v = util::set( 4, 3, 2, 1 );

	util::vec _u = util::load( v );
	_u = _u + _v;

	v = util::store( _u );

	BOOST_REQUIRE( 5 == v.x );
	BOOST_REQUIRE( 5 == v.y );
	BOOST_REQUIRE( 5 == v.z );
	BOOST_REQUIRE( 5 == v.w );
}

BOOST_AUTO_TEST_CASE( clamp )
{
	BOOST_REQUIRE( util::clamp( -3, 0, 5 ) == 0 );
	BOOST_REQUIRE( util::clamp( 0, 0, 5 ) == 0 );
	BOOST_REQUIRE( util::clamp( 2, 0, 5 ) == 2 );
	BOOST_REQUIRE( util::clamp( 5, 0, 5 ) == 5 );
	BOOST_REQUIRE( util::clamp( 7, 0, 5 ) == 5 );
}

BOOST_AUTO_TEST_CASE( homogenize )
{
	util::float4 v( 1.0f, 1.0f, 1.0f, 2.0f );
	util::vec _v = util::load( v );
	_v = util::homogenize( _v );
	v = util::store( _v );

	BOOST_REQUIRE( v.x == 0.5f );
	BOOST_REQUIRE( v.w == 1.0f );
}

BOOST_AUTO_TEST_CASE( powerOf2 )
{
	BOOST_REQUIRE( util::powerOf2( 1 ) );
	BOOST_REQUIRE( util::powerOf2( 2 ) );
	BOOST_REQUIRE( util::powerOf2( 4 ) );
	BOOST_REQUIRE( util::powerOf2( 2048 ) );
	BOOST_REQUIRE( util::powerOf2( 65536 ) );
	BOOST_REQUIRE( util::powerOf2( 2048 * 2048 ) );

	BOOST_REQUIRE( ! util::powerOf2( 0 ) );
	BOOST_REQUIRE( ! util::powerOf2( 3 ) );
	BOOST_REQUIRE( ! util::powerOf2( 13 ) );
	BOOST_REQUIRE( ! util::powerOf2( 26 ) );
	BOOST_REQUIRE( ! util::powerOf2( 48 ) );
}

// TODO: Rework packInts and test!

BOOST_AUTO_TEST_SUITE_END()