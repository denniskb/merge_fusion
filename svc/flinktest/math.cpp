#include <boost/test/auto_unit_test.hpp>

#include <flink/math.h>



BOOST_AUTO_TEST_SUITE( math )

BOOST_AUTO_TEST_CASE( homogenize )
{
	flink::float4 v( 1.0f, 1.0f, 1.0f, 2.0f );
	flink::vec _v = flink::load( v );
	_v = flink::homogenize( _v );
	v = flink::store( _v );

	BOOST_REQUIRE( v.x == 0.5f );
	BOOST_REQUIRE( v.w == 1.0f );
}

BOOST_AUTO_TEST_CASE( clamp )
{
	BOOST_REQUIRE( flink::clamp( -3, 0, 5 ) == 0 );
	BOOST_REQUIRE( flink::clamp( 0, 0, 5 ) == 0 );
	BOOST_REQUIRE( flink::clamp( 2, 0, 5 ) == 2 );
	BOOST_REQUIRE( flink::clamp( 5, 0, 5 ) == 5 );
	BOOST_REQUIRE( flink::clamp( 7, 0, 5 ) == 5 );
}

BOOST_AUTO_TEST_SUITE_END()