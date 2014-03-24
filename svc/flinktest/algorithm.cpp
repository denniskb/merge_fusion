#include <boost/test/auto_unit_test.hpp>

#include <cstdlib>

#include <flink/algorithm.h>



BOOST_AUTO_TEST_SUITE( algorithm )

BOOST_AUTO_TEST_CASE( radix_sort )
{
	flink::vector< unsigned > data( 100 );

	for( int i = 0; i < 100; i++ )
		data[ i ] = rand();

	flink::radix_sort( data );

	for( int i = 0; i < 99; i++ )
		BOOST_REQUIRE( data[ i ] < data[ i + 1 ] );
}

BOOST_AUTO_TEST_CASE( radix_sort2 )
{
	flink::vector< unsigned > keys( 100 );
	flink::vector< unsigned > values( 100 );

	for( int i = 0; i < 100; i++ )
		keys[ i ] = values[ i ] = rand();

	flink::radix_sort( keys, values );

	for( int i = 0; i < 99; i++ )
	{
		BOOST_REQUIRE( keys[ i ] < keys[ i + 1 ] );
		BOOST_REQUIRE( values[ i ] < values[ i + 1 ] );
	}
}



BOOST_AUTO_TEST_CASE( remove_dups )
{
	flink::vector< unsigned > test;
	test.push_back( 1 );
	test.push_back( 1 );
	test.push_back( 2 );

	flink::remove_dups( test );

	BOOST_REQUIRE( test.size() == 2 );
	BOOST_REQUIRE( test[ 0 ] == 1 );
	BOOST_REQUIRE( test[ 1 ] == 2 );
}

BOOST_AUTO_TEST_CASE( remove_value )
{
	flink::vector< int > test;
	test.push_back( 1 );
	test.push_back( 1 );
	test.push_back( 2 );

	flink::remove_value( test, 1 );

	BOOST_REQUIRE( test.size() == 1 );
	BOOST_REQUIRE( test[ 0 ] == 2 );
}



BOOST_AUTO_TEST_CASE( intersection_size )
{
	flink::vector< int > a, b;
	BOOST_REQUIRE( flink::intersection_size( a.cbegin(), a.cend(), b.cbegin(), b.cend() ) == 0 );

	a.push_back( 1 );
	BOOST_REQUIRE( flink::intersection_size( a.cbegin(), a.cend(), b.cbegin(), b.cend() ) == 0 );

	b.push_back( 1 );
	BOOST_REQUIRE( flink::intersection_size( a.cbegin(), a.cend(), b.cbegin(), b.cend() ) == 1 );

	b.push_back( 3 );
	BOOST_REQUIRE( flink::intersection_size( a.cbegin(), a.cend(), b.cbegin(), b.cend() ) == 1 );

	a.push_back( 3 );
	BOOST_REQUIRE( flink::intersection_size( a.cbegin(), a.cend(), b.cbegin(), b.cend() ) == 2 );

	a.push_back( 7 );
	BOOST_REQUIRE( flink::intersection_size( a.cbegin(), a.cend(), b.cbegin(), b.cend() ) == 2 );
}

BOOST_AUTO_TEST_SUITE_END()