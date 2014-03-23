#include <boost/test/auto_unit_test.hpp>

#include <flink/flat_map.h>



BOOST_AUTO_TEST_SUITE( flat_map )

BOOST_AUTO_TEST_CASE( ctor )
{
	flink::flat_map< int, int > test;
	BOOST_REQUIRE( test.size() == 0 );

	flink::vector< int > keys, values;
	keys.push_back( 2 );
	keys.push_back( 3 );
	values.push_back( 13 );
	values.push_back( -7 );
	test = flink::flat_map< int, int >( std::move( keys ), std::move( values ) );
	
	BOOST_REQUIRE( test.size() == 2 );
	BOOST_REQUIRE( test.keys()[ 0 ] == 2 );
	BOOST_REQUIRE( test.keys()[ 1 ] == 3 );
	BOOST_REQUIRE( test.values()[ 0 ] == 13 );
	BOOST_REQUIRE( test.values()[ 1 ] == -7 );

	test.clear();
	BOOST_REQUIRE( test.size() == 0 );
}

BOOST_AUTO_TEST_CASE( insert )
{
	flink::vector< int > keys, values;
	keys.push_back( 2 );
	keys.push_back( 3 );
	values.push_back( 13 );
	values.push_back( -7 );

	flink::flat_map< int, int > a, b( std::move( keys ), std::move( values ) );
	a.insert( b );

	BOOST_REQUIRE( a.size() == 2 );
	BOOST_REQUIRE( a.keys()[ 0 ] == 2 );
	BOOST_REQUIRE( a.keys()[ 1 ] == 3 );
	BOOST_REQUIRE( a.values()[ 0 ] == 13 );
	BOOST_REQUIRE( a.values()[ 1 ] == -7 );

	a.insert( b );
	BOOST_REQUIRE( a.size() == 4 );
	BOOST_REQUIRE( a.keys()[ 1 ] == 2 );
	BOOST_REQUIRE( a.values()[ 3 ] == -7 );
}

BOOST_AUTO_TEST_SUITE_END()