#include <cstdlib>

#include <boost/test/auto_unit_test.hpp>

#include <kifi/util/algorithm.h>
#include <kifi/util/numeric.h>

using namespace kifi;



BOOST_AUTO_TEST_SUITE( util_test )
BOOST_AUTO_TEST_SUITE( algorithm_test )

BOOST_AUTO_TEST_CASE( radix_sort )
{
	std::vector< unsigned > keys( 100 );
	std::vector< unsigned > tmp( keys.size() );

	for( int i = 0; i < keys.size(); i++ )
		keys[ i ] = std::rand();

	util::radix_sort( keys.data(), keys.data() + keys.size(), tmp.data() );

	for( int i = 0; i < keys.size() - 1; i++ )
		BOOST_REQUIRE( keys[ i ] < keys[ i + 1 ] );
}

BOOST_AUTO_TEST_CASE( radix_sort2 )
{
	std::vector< unsigned > keys( 100 );
	std::vector< unsigned > values( keys.size() );
	std::vector< unsigned > tmp( keys.size() + values.size() );

	for( int i = 0; i < keys.size(); i++ )
		keys[ i ] = values[ i ] = std::rand();

	util::radix_sort( keys.data(), keys.data() + keys.size(), values.data(), tmp.data() );

	for( int i = 0; i < keys.size() - 1; i++ )
		BOOST_REQUIRE( keys[ i ] < keys[ i + 1 ] );

	for( int i = 0; i < keys.size(); i++ )
		BOOST_REQUIRE( keys[ i ] == values[ i ] );
}

BOOST_AUTO_TEST_CASE( set_union )
{
	{
		int const k1[] = {  0 };
		int const v1[] = { -1 };
		int const k2[] = {  1 };
		int const v2[] = {  7 };

		int r1[] = { -1, -1 };
		int r2[] = { -1, -1 };

		util::set_union
		(
			k1, k1+1,
			k2, k2+1,
			v1, v2,
			r1, r2
		);
		BOOST_REQUIRE(  0 == r1[ 0 ] );
		BOOST_REQUIRE(  1 == r1[ 1 ] );
		BOOST_REQUIRE( -1 == r2[ 0 ] );
		BOOST_REQUIRE(  7 == r2[ 1 ] );
	}

	{
		int const k1[] = {  7 };
		int const v1[] = { -1 };
		int const k2[] = {  7 };
		int const v2[] = {  3 };

		int r1[] = { -1, -1 };
		int r2[] = { -1, -1 };

		util::set_union
		(
			k1, k1+1,
			k2, k2+1,
			v1, v2,
			r1, r2
		);
		BOOST_REQUIRE(  7 == r1[ 0 ] );
		BOOST_REQUIRE( -1 == r1[ 1 ] );
		BOOST_REQUIRE( -1 == r2[ 0 ] );
		BOOST_REQUIRE( -1 == r2[ 1 ] );
	}

	{
		int const k1[] = { 0, 1, 3, 55 };
		int const v1[] = { 4, 9, 0, 27 };
		int const k2[] = { 0, 5, 27, 55, 88 };
		int const v2[] = { 3, 1, 17,  0,  7 };

		int r1[ 7 ]; std::fill( r1, r1+7, -1 );
		int r2[ 7 ]; std::fill( r2, r2+7, -1 );

		util::set_union
		(
			k1, k1+4,
			k2, k2+5,
			v1, v2,
			r1, r2
		);
		BOOST_REQUIRE(  0 == r1[ 0 ] );
		BOOST_REQUIRE(  1 == r1[ 1 ] );
		BOOST_REQUIRE(  3 == r1[ 2 ] );
		BOOST_REQUIRE(  5 == r1[ 3 ] );
		BOOST_REQUIRE( 27 == r1[ 4 ] );
		BOOST_REQUIRE( 55 == r1[ 5 ] );
		BOOST_REQUIRE( 88 == r1[ 6 ] );

		BOOST_REQUIRE(  4 == r2[ 0 ] );
		BOOST_REQUIRE(  9 == r2[ 1 ] );
		BOOST_REQUIRE(  0 == r2[ 2 ] );
		BOOST_REQUIRE(  1 == r2[ 3 ] );
		BOOST_REQUIRE( 17 == r2[ 4 ] );
		BOOST_REQUIRE( 27 == r2[ 5 ] );
		BOOST_REQUIRE(  7 == r2[ 6 ] );
	}
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()