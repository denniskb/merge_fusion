#include <boost/test/auto_unit_test.hpp>

#include <cstdlib>
#include <vector>

#include <server/radix_sort.h>



BOOST_AUTO_TEST_SUITE( radix_sort )

BOOST_AUTO_TEST_CASE( sort )
{
	std::vector< unsigned > data( 100 );
	std::vector< unsigned > tmp;

	for( int i = 0; i < 100; i++ )
		data[ i ] = rand();

	kppl::radix_sort( data, tmp );

	for( int i = 0; i < 99; i++ )
		BOOST_REQUIRE( data[ i ] < data[ i + 1 ] );
}

BOOST_AUTO_TEST_SUITE_END()