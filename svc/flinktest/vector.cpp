#include <boost/test/auto_unit_test.hpp>

#include <flink/vector.h>



BOOST_AUTO_TEST_SUITE( vector )

BOOST_AUTO_TEST_CASE( ctor )
{
	flink::vector< int > v;
	BOOST_REQUIRE( v.begin() == nullptr );
	BOOST_REQUIRE( v.cbegin() == nullptr );
	BOOST_REQUIRE( v.size() == 0 );
	BOOST_REQUIRE( v.capacity() == 0 );

	flink::vector< int > v2( 1 );
	BOOST_REQUIRE( v2.size() == 1 );
	BOOST_REQUIRE( v2.capacity() == 1 );

	v2[ 0 ] = 7;
	BOOST_REQUIRE( v2[ 0 ] == 7 );
}

BOOST_AUTO_TEST_CASE( copy_move_assign )
{
	// move
	flink::vector< int > v( flink::vector< int >( 1 ) );
	BOOST_REQUIRE( v.size() == 1 );
	BOOST_REQUIRE( v.capacity() == 1 );

	// copy
	v[ 0 ] = 7;
	flink::vector< int > v2( v );
	BOOST_REQUIRE( v2.size() == v.size() );
	BOOST_REQUIRE( v2.capacity() == v.capacity() );
	BOOST_REQUIRE( v2[ 0 ] == 7 );
	BOOST_REQUIRE( v.size() == 1 );
	BOOST_REQUIRE( v[ 0 ] == 7 );

	// assign
	v[ 0 ] = 2;
	v2 = v;
	BOOST_REQUIRE( v2.size() == v.size() );
	BOOST_REQUIRE( v2.capacity() == v.capacity() );
	BOOST_REQUIRE( v2[ 0 ] == 2 );

	// move assign
	v = flink::vector< int >( 2 );
	BOOST_REQUIRE( v.size() == 2 );
	BOOST_REQUIRE( v.capacity() == 2 );
}

BOOST_AUTO_TEST_CASE( resize )
{
	flink::vector< int > v;
	v.resize( 1 );
	BOOST_REQUIRE( v.size() == 1 );
	BOOST_REQUIRE( v.capacity() == 1 );
}

BOOST_AUTO_TEST_CASE( swap )
{
	using std::swap;

	flink::vector< int > v;
	flink::vector< int > v2( 1 );
	swap( v, v2 );

	BOOST_REQUIRE( v.size() == 1 );
	BOOST_REQUIRE( v.capacity() == 1 );
	BOOST_REQUIRE( v2.size() == 0 );
	BOOST_REQUIRE( v2.capacity() == 0 );
}

BOOST_AUTO_TEST_CASE( begin_end )
{
	flink::vector< int > v( 1 );
	BOOST_REQUIRE( v.begin() == & v[ 0 ] );
	BOOST_REQUIRE( v.end() == & v[ 0 ] + 1 );

	BOOST_REQUIRE( v.cbegin() == & v[ 0 ] );
	BOOST_REQUIRE( v.cend() == & v[ 0 ] + 1 );
}

BOOST_AUTO_TEST_CASE( clear )
{
	flink::vector< int > v( 1 );
	v.clear();

	BOOST_REQUIRE( v.size() == 0 );
}

BOOST_AUTO_TEST_CASE( push_back )
{
	flink::vector< int > v;
	v.push_back( 1 );

	BOOST_REQUIRE( v.size() == 1 );
	BOOST_REQUIRE( v[ 0 ] == 1 );
}

BOOST_AUTO_TEST_SUITE_END()