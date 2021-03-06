#include <vector>

#include <boost/test/auto_unit_test.hpp>

#include <kifi/util/flat_map.h>
#include <kifi/util/iterator.h>

using namespace kifi;



BOOST_AUTO_TEST_SUITE( util_test )
BOOST_AUTO_TEST_SUITE( flat_map_test )

BOOST_AUTO_TEST_CASE( ctor )
{
	util::flat_map< int, int > a;

	BOOST_REQUIRE( 0 == a.size() );
}

BOOST_AUTO_TEST_CASE( clear )
{
	util::flat_map< int, int > a;
	int k = 7;
	int v = 3;

	a.insert( &k, &k+1, &v );
	BOOST_REQUIRE( 1 == a.size() );

	a.clear();
	BOOST_REQUIRE( 0 == a.size() );
}

BOOST_AUTO_TEST_CASE( insert )
{
	{
		util::flat_map< int, int > a;
		int k[] = { 7 };
		int v[] = { 3 };

		a.insert( k, k+1, v );
		// a.keys   == { 7 }
		// a.values == { 3 }

		BOOST_REQUIRE( 1 == a.size() );
		BOOST_REQUIRE( 7 == a.keys_cbegin()[ 0 ] );
		BOOST_REQUIRE( 3 == a.values_cbegin()[ 0 ] );

		a.insert( k, k+1, v );
		// a should not change

		BOOST_REQUIRE( 1 == a.size() );
		BOOST_REQUIRE( 7 == a.keys_cbegin()[ 0 ] );
		BOOST_REQUIRE( 3 == a.values_cbegin()[ 0 ] );
	}

	{
		util::flat_map< int, int > a;
		int k[]  = { 7 };
		int v[]  = { 117 };

		int k2[] = {   3,   7,  55 };
		int v2[] = { 449, 551, 666 };

		a.insert( k, k+1, v );
		// a.keys   == {   7 }
		// a.values == { 117 }

		a.insert( k2, k2+3, v2 );
		// a.keys   == {   3,   7,  55 }
		// a.values == { 449, 117, 666 }

		BOOST_REQUIRE( a.size() == 3 );

		BOOST_REQUIRE(  3 == a.keys_cbegin()[ 0 ] );
		BOOST_REQUIRE(  7 == a.keys_cbegin()[ 1 ] );
		BOOST_REQUIRE( 55 == a.keys_cbegin()[ 2 ] );

		BOOST_REQUIRE( 449 == a.values_cbegin()[ 0 ] );
		BOOST_REQUIRE( 117 == a.values_cbegin()[ 1 ] );
		BOOST_REQUIRE( 666 == a.values_cbegin()[ 2 ] );
	}
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()