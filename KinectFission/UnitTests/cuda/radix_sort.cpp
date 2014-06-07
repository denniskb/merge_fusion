#include <boost/test/auto_unit_test.hpp>

#include <kifi/cuda/radix_sort.h>
#include <kifi/cuda/timer.h>
#include <kifi/cuda/vector.h>

using namespace kifi;



BOOST_AUTO_TEST_SUITE( cuda_test )
BOOST_AUTO_TEST_SUITE( radix_sort_test )

BOOST_AUTO_TEST_CASE( test )
{
	std::vector< unsigned > test( 10 * 1000 * 1000 );
	cuda::vector< unsigned > data( test.size() );
	cuda::vector< unsigned > tmp( data.capacity() );

	for( int i = 0; i < test.size(); i++ )
		test[ i ] = rand() % 16;

	cuda::copy( data, test );
	tmp.resize( data.capacity() );

	cuda::timer t;
	cuda::radix_sort( data.data(), (int) data.capacity(), tmp.data() );

	float time = t.record_time( "test" );
 	printf( "%.2fms (%.1fGB/s)\n", time, data.capacity() / time * 4000 / 1024 / 1024 / 1024 );

	cuda::copy( test, tmp );
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()