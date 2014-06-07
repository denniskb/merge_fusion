#include <boost/test/auto_unit_test.hpp>

#include <kifi/util/numeric.h>

using namespace kifi;



BOOST_AUTO_TEST_SUITE( util_test )
BOOST_AUTO_TEST_SUITE( numeric_test )

BOOST_AUTO_TEST_CASE( partial_sum_exclusive )
{
	int a[]             = { 1, 3, 3, 7,  0 };
	int const scanOfA[] = { 0, 1, 4, 7, 14 };
	
	util::partial_sum_exclusive( a, a + 5, a );

	for( int i = 0; i < 5; i++ )
		BOOST_REQUIRE( scanOfA[ i ] == a[ i ] );
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()