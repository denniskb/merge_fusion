#include <boost/test/auto_unit_test.hpp>

#include <cstdlib>
#include <vector>

#include <reference/radix_sort.h>



BOOST_AUTO_TEST_SUITE( radix_sort )

BOOST_AUTO_TEST_CASE( sort )
{
	svc::vector< unsigned > data( 100 );

	for( int i = 0; i < 100; i++ )
		data[ i ] = rand();

	svc::radix_sort( data );

	for( int i = 0; i < 99; i++ )
		BOOST_REQUIRE( data[ i ] < data[ i + 1 ] );
}

BOOST_AUTO_TEST_SUITE_END()