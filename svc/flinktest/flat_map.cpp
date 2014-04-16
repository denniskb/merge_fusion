#include <boost/test/auto_unit_test.hpp>

#include <flink/flat_map.h>
#include <flink/vector.h>



BOOST_AUTO_TEST_SUITE( flat_map )

BOOST_AUTO_TEST_CASE( ctor )
{
	flink::flat_map< int, int > a;

	BOOST_REQUIRE( a.size() == 0 );
}

BOOST_AUTO_TEST_CASE( clear )
{
	flink::flat_map< int, int > a;
	int k = 7;

	a.merge_unique( &k, &k+1, 3 );
	BOOST_REQUIRE( a.size() == 1 );

	a.clear();
	BOOST_REQUIRE( a.size() == 0 );
}

BOOST_AUTO_TEST_CASE( merge_unique )
{
	{
		flink::flat_map< int, int > a;
		int k = 7;

		a.merge_unique( &k, &k+1, 3 );
		BOOST_REQUIRE( a.size() == 1 );
		BOOST_REQUIRE( a.keys_first()[ 0 ] == 7 );
		BOOST_REQUIRE( a.values_first()[ 0 ] == 3 );

		a.merge_unique( &k, &k+1, 3 );
		BOOST_REQUIRE( a.size() == 1 );
		BOOST_REQUIRE( a.keys_first()[ 0 ] == 7 );
		BOOST_REQUIRE( a.values_first()[ 0 ] == 3 );

		int k2[] = { 0, 3, 7, 55 };

		a.merge_unique( k2, k2+4, 9 );
		BOOST_REQUIRE( a.size() == 4 );
		BOOST_REQUIRE( a.keys_first()[ 0 ] ==  0 );
		BOOST_REQUIRE( a.keys_first()[ 1 ] ==  3 );
		BOOST_REQUIRE( a.keys_first()[ 2 ] ==  7 );
		BOOST_REQUIRE( a.keys_first()[ 3 ] == 55 );
		BOOST_REQUIRE( a.values_first()[ 0 ] == 9 );
		BOOST_REQUIRE( a.values_first()[ 1 ] == 9 );
		BOOST_REQUIRE( a.values_first()[ 2 ] == 3 );
		BOOST_REQUIRE( a.values_first()[ 3 ] == 9 );
	}
}

BOOST_AUTO_TEST_SUITE_END()