#include <boost/test/auto_unit_test.hpp>

#include <cuda/radix_sort.h>
#include <cuda/timer.h>
#include <cuda/vector.h>



BOOST_AUTO_TEST_SUITE( radix_sort )

BOOST_AUTO_TEST_CASE( test )
{
	std::vector< unsigned > test( 1 * 100 * 1000 );
	svcu::vector< unsigned > data( test.size() );
	svcu::vector< unsigned > tmp( data.capacity() );

	for( int i = 0; i < test.size(); i++ )
		test[ i ] = rand() % 7919;

	svcu::copy( data, test );
	tmp.resize( data.capacity() );

	//svcu::timer t;
	svcu::radix_sort( data.data(), (int) data.capacity(), tmp.data() );

	//float time = t.record_time( "test" );
	//printf( "%.2fms (%.1fGB/s)\n", time, data.capacity() / time * 4000 / 1024 / 1024 / 1024 );
}

BOOST_AUTO_TEST_SUITE_END()