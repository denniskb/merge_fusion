#include <boost/test/auto_unit_test.hpp>

#include <reference/algorithm.h>



BOOST_AUTO_TEST_SUITE( algorithm )

BOOST_AUTO_TEST_CASE( exclusive_scan )
{
	int a[] = { 1, 3, 3, 7,  0,  9 };
	int b[] = { 0, 1, 4, 7, 14, 14 };

	svc::exclusive_scan( a, a + 6 );

	for( int i = 0; i < 6; i++ )
		BOOST_REQUIRE( a[ i ] == b[ i ] );
}

BOOST_AUTO_TEST_CASE( inclusive_scan )
{
	int a[] = { 1, 3, 3,  7,  0,  9 };
	int b[] = { 1, 4, 7, 14, 14, 23 };

	svc::inclusive_scan( a, a + 6 );

	for( int i = 0; i < 6; i++ )
		BOOST_REQUIRE( a[ i ] == b[ i ] );
}



BOOST_AUTO_TEST_CASE( intersection_size )
{
	std::vector< int > a, b;
	BOOST_REQUIRE( svc::intersection_size
	(
		a.data(), a.data() + a.size(), 
		b.data(), b.data() + b.size() 
	) == 0 );

	a.push_back( 1 );
	BOOST_REQUIRE( svc::intersection_size
	(
		a.data(), a.data() + a.size(), 
		b.data(), b.data() + b.size() 
	) == 0 );

	b.push_back( 1 );
	BOOST_REQUIRE( svc::intersection_size
	(
		a.data(), a.data() + a.size(), 
		b.data(), b.data() + b.size() 
	) == 1 );

	b.push_back( 3 );
	BOOST_REQUIRE( svc::intersection_size
	(
		a.data(), a.data() + a.size(), 
		b.data(), b.data() + b.size() 
	) == 1 );

	a.push_back( 3 );
	BOOST_REQUIRE( svc::intersection_size
	(
		a.data(), a.data() + a.size(), 
		b.data(), b.data() + b.size() 
	) == 2 );

	a.push_back( 7 );
	BOOST_REQUIRE( svc::intersection_size
	(
		a.data(), a.data() + a.size(), 
		b.data(), b.data() + b.size() 
	) == 2 );
}



BOOST_AUTO_TEST_CASE( reduce )
{
	int a[] = { 1, 3, 3,  7,  0,  9 };

	int sum = svc::reduce( a, a + 6 );

	BOOST_REQUIRE( 23 == sum );
}



BOOST_AUTO_TEST_CASE( set_union_backward )
{
	{
		int a[] = { 0 };
		int b[] = { 1 };
		int c[] = { -1, -1 };

		svc::set_union_backward( a, a+1, b, b+1, c+2 );
		BOOST_REQUIRE( c[ 0 ] == 0 );
		BOOST_REQUIRE( c[ 1 ] == 1 );

		c[ 0 ] = c[ 1 ] = -1;

		svc::set_union_backward( b, b+1, a, a+1, c+2 );
		BOOST_REQUIRE( c[ 0 ] == 0 );
		BOOST_REQUIRE( c[ 1 ] == 1 );

		a[ 0 ] = b[ 0 ] = 7;
		c[ 0 ] = c[ 1 ] = -1;

		svc::set_union_backward( a, a+1, b, b+1, c+2 );
		BOOST_REQUIRE( c[ 0 ] == -1 );
		BOOST_REQUIRE( c[ 1 ] ==  7 );
	}

	{
		int a[] = { 0, 1, 3, 55 };
		int b[] = { 0, 5, 27, 55, 88 };
		int c[ 7 ]; std::fill( c, c+7, -1 );

		svc::set_union_backward( a, a+4, b, b+5, c+7 );
		BOOST_REQUIRE( c[ 0 ] ==  0 );
		BOOST_REQUIRE( c[ 1 ] ==  1 );
		BOOST_REQUIRE( c[ 2 ] ==  3 );
		BOOST_REQUIRE( c[ 3 ] ==  5 );
		BOOST_REQUIRE( c[ 4 ] == 27 );
		BOOST_REQUIRE( c[ 5 ] == 55 );
		BOOST_REQUIRE( c[ 6 ] == 88 );
	}
}

BOOST_AUTO_TEST_CASE( set_union_backward2 )
{
	{
		int k1[] = {  0 };
		int v1[] = { -1 };

		int k2[] = {  1 };

		int r1[] = { -1, -1 };
		int r2[] = { -1, -1 };

		int v = 7;
		svc::set_union_backward
		(
			k1, k1+1, v1+1, 
			k2, k2+1, v, 
			r1+2, r2+2
		);
		BOOST_REQUIRE( r1[ 0 ] == 0 );
		BOOST_REQUIRE( r1[ 1 ] == 1 );
		BOOST_REQUIRE( r2[ 0 ] == -1 );
		BOOST_REQUIRE( r2[ 1 ] ==  7 );

		k1[ 0 ] = k2[ 0 ] = 7;
		r1[ 0 ] = r1[ 1 ] = -1;
		r2[ 0 ] = r2[ 1 ] = -1;

		v = 3;
		svc::set_union_backward
		(
			k1, k1+1, v1+1, 
			k2, k2+1, v, 
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

		int v = 3;
		svc::set_union_backward
		(
			k1, k1+4, v1+4, 
			k2, k2+5, v, 
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



BOOST_AUTO_TEST_CASE( radix_sort )
{
	std::vector< unsigned > data( 100 );
	std::vector< char > tmp;

	for( int i = 0; i < 100; i++ )
		data[ i ] = rand();

	svc::radix_sort( data.begin(), data.end(), tmp );

	for( int i = 0; i < 99; i++ )
		BOOST_REQUIRE( data[ i ] < data[ i + 1 ] );
}

BOOST_AUTO_TEST_CASE( radix_sort2 )
{
	std::vector< unsigned > keys( 100 );
	std::vector< unsigned > values( 100 );
	std::vector< char > tmp;

	for( int i = 0; i < 100; i++ )
		keys[ i ] = values[ i ] = rand();

	svc::radix_sort( keys.begin(), keys.end(), values.begin(), tmp );

	for( int i = 0; i < 99; i++ )
	{
		BOOST_REQUIRE( keys[ i ] < keys[ i + 1 ] );
		BOOST_REQUIRE( values[ i ] < values[ i + 1 ] );
	}
}



BOOST_AUTO_TEST_CASE( unique )
{
	{
		int a[] = { 7 };

		size_t newSize = svc::unique( a, a + 1 );

		BOOST_REQUIRE( newSize == 1 );
		BOOST_REQUIRE( a[ 0 ] == 7 );
	}

	{
		int a[] = { 2, 2 };

		size_t newSize = svc::unique( a, a + 2 );

		BOOST_REQUIRE( newSize == 1 );
		BOOST_REQUIRE( a[ 0 ] == 2 );
	}

	{
		int a[] = { 1, 1, 2 };

		size_t newSize = svc::unique( a, a + 3 );

		BOOST_REQUIRE( newSize == 2 );
		BOOST_REQUIRE( a[ 0 ] == 1 );
		BOOST_REQUIRE( a[ 1 ] == 2 );
	}
}

BOOST_AUTO_TEST_SUITE_END()