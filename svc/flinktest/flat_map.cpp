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

BOOST_AUTO_TEST_CASE( merge_unique )
{
	flink::vector< int > keys1, values1, keys2;
	flink::flat_map< int, int >test( keys1, values1 );

	test.merge_unique( keys2.cbegin(), keys2.cend(), 0 );
	BOOST_REQUIRE( test.size() == 0 );

	keys1.push_back( 1 ); values1.push_back( -7 );
	test = flink::flat_map< int, int >( keys1, values1 );
	
	test.merge_unique( keys2.cbegin(), keys2.cend(), 0 );
	BOOST_REQUIRE( test.size() == 1 );
	BOOST_REQUIRE( test.keys()[ 0 ] == 1 );
	BOOST_REQUIRE( test.values()[ 0 ] == -7 );

	test.clear();
	keys2.push_back( 1 );

	test.merge_unique( keys2.cbegin(), keys2.cend(), 5 );
	BOOST_REQUIRE( test.size() == 1 );
	BOOST_REQUIRE( test.keys()[ 0 ] == 1 );
	BOOST_REQUIRE( test.values()[ 0 ] == 5 );

	test = flink::flat_map< int, int >( keys1, values1 );
	test.merge_unique( keys2.cbegin(), keys2.cend(), 5 );

	BOOST_REQUIRE( test.size() == 1 );
	BOOST_REQUIRE( test.keys()[ 0 ] == 1 );
	BOOST_REQUIRE( test.values()[ 0 ] == -7 );

	keys2.push_back( 3 );
	test.merge_unique( keys2.cbegin(), keys2.cend(), 5 );
	BOOST_REQUIRE( test.size() == 2 );
	BOOST_REQUIRE( test.keys()[ 0 ] == 1 );
	BOOST_REQUIRE( test.keys()[ 1 ] == 3 );
	BOOST_REQUIRE( test.values()[ 0 ] == -7 );
	BOOST_REQUIRE( test.values()[ 1 ] ==  5 );
}

BOOST_AUTO_TEST_SUITE_END()