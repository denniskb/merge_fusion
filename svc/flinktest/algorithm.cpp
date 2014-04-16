#include <boost/test/auto_unit_test.hpp>

#include <flink/algorithm.h>



BOOST_AUTO_TEST_SUITE( algorithm )

BOOST_AUTO_TEST_CASE( radix_sort )
{
	flink::vector< unsigned > data( 100 );
	flink::vector< char > tmp;

	for( int i = 0; i < 100; i++ )
		data[ i ] = rand();

	flink::radix_sort( data.begin(), data.size(), tmp );

	for( int i = 0; i < 99; i++ )
		BOOST_REQUIRE( data[ i ] < data[ i + 1 ] );
}

BOOST_AUTO_TEST_CASE( radix_sort2 )
{
	flink::vector< unsigned > keys( 100 );
	flink::vector< unsigned > values( 100 );
	flink::vector< char > tmp;

	for( int i = 0; i < 100; i++ )
		keys[ i ] = values[ i ] = rand();

	flink::radix_sort( keys.begin(), values.begin(), keys.size(), tmp );

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



BOOST_AUTO_TEST_CASE( merge_unique_backward )
{
	{
		int a[] = { 0 };
		int b[] = { 1 };
		int c[] = { -1, -1 };

		flink::merge_unique_backward( a, a+1, b, b+1, c+2 );
		BOOST_REQUIRE( c[ 0 ] == 0 );
		BOOST_REQUIRE( c[ 1 ] == 1 );

		c[ 0 ] = c[ 1 ] = -1;

		flink::merge_unique_backward( b, b+1, a, a+1, c+2 );
		BOOST_REQUIRE( c[ 0 ] == 0 );
		BOOST_REQUIRE( c[ 1 ] == 1 );

		a[ 0 ] = b[ 0 ] = 7;
		c[ 0 ] = c[ 1 ] = -1;

		flink::merge_unique_backward( a, a+1, b, b+1, c+2 );
		BOOST_REQUIRE( c[ 0 ] == -1 );
		BOOST_REQUIRE( c[ 1 ] ==  7 );
	}

	{
		int a[] = { 0, 1, 3, 55 };
		int b[] = { 0, 5, 27, 55, 88 };
		int c[ 7 ]; std::fill( c, c+7, -1 );

		flink::merge_unique_backward( a, a+4, b, b+5, c+7 );
		BOOST_REQUIRE( c[ 0 ] ==  0 );
		BOOST_REQUIRE( c[ 1 ] ==  1 );
		BOOST_REQUIRE( c[ 2 ] ==  3 );
		BOOST_REQUIRE( c[ 3 ] ==  5 );
		BOOST_REQUIRE( c[ 4 ] == 27 );
		BOOST_REQUIRE( c[ 5 ] == 55 );
		BOOST_REQUIRE( c[ 6 ] == 88 );
	}
}

BOOST_AUTO_TEST_CASE( merge_unique_backward2 )
{
	{
		int k1[] = {  0 };
		int v1[] = { -1 };

		int k2[] = {  1 };

		int r1[] = { -1, -1 };
		int r2[] = { -1, -1 };

		flink::merge_unique_backward
		(
			k1, k1+1, v1+1, 
			k2, k2+1, 7, 
			r1+2, r2+2
		);
		BOOST_REQUIRE( r1[ 0 ] == 0 );
		BOOST_REQUIRE( r1[ 1 ] == 1 );
		BOOST_REQUIRE( r2[ 0 ] == -1 );
		BOOST_REQUIRE( r2[ 1 ] ==  7 );

		k1[ 0 ] = k2[ 0 ] = 7;
		r1[ 0 ] = r1[ 1 ] = -1;
		r2[ 0 ] = r2[ 1 ] = -1;

		flink::merge_unique_backward
		(
			k1, k1+1, v1+1, 
			k2, k2+1, 3, 
			r1+2, r2+2
		);
		BOOST_REQUIRE( r1[ 0 ] == -1 );
		BOOST_REQUIRE( r1[ 1 ] ==  7 );
		BOOST_REQUIRE( r2[ 0 ] == -1 );
		BOOST_REQUIRE( r2[ 1 ] == -1 );
	}

	{
		int k1[] = { 0, 1, 3, 55 };
		int v1[] = { 4, 9, 0, 27 };

		int k2[] = { 0, 5, 27, 55, 88 };

		int r1[ 7 ]; std::fill( r1, r1+7, -1 );
		int r2[ 7 ]; std::fill( r2, r2+7, -1 );

		flink::merge_unique_backward
		(
			k1, k1+4, v1+4, 
			k2, k2+5, 3, 
			r1+7, r2+7
		);
		BOOST_REQUIRE( r1[ 0 ] ==  0 );
		BOOST_REQUIRE( r1[ 1 ] ==  1 );
		BOOST_REQUIRE( r1[ 2 ] ==  3 );
		BOOST_REQUIRE( r1[ 3 ] ==  5 );
		BOOST_REQUIRE( r1[ 4 ] == 27 );
		BOOST_REQUIRE( r1[ 5 ] == 55 );
		BOOST_REQUIRE( r1[ 6 ] == 88 );

		BOOST_REQUIRE( r2[ 0 ] ==  4 );
		BOOST_REQUIRE( r2[ 1 ] ==  9 );
		BOOST_REQUIRE( r2[ 2 ] ==  0 );
		BOOST_REQUIRE( r2[ 3 ] ==  3 );
		BOOST_REQUIRE( r2[ 4 ] ==  3 );
		BOOST_REQUIRE( r2[ 5 ] == 27 );
		BOOST_REQUIRE( r2[ 6 ] ==  3 );
	}
}

BOOST_AUTO_TEST_SUITE_END()