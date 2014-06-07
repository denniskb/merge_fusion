#include <boost/test/auto_unit_test.hpp>

#include <algorithm>
#include <numeric>
#include <vector>

#include <kifi/cuda/scan.h>
#include <kifi/cuda/timer.h>
#include <kifi/cuda/vector.h>

#include <kifi/util/algorithm.h>
#include <kifi/util/numeric.h>

using namespace kifi;



BOOST_AUTO_TEST_SUITE( cuda_test )
BOOST_AUTO_TEST_SUITE( scan_test )

BOOST_AUTO_TEST_CASE( segmented_inclusive_scan )
{
	int const segmentSize = 256;

	std::vector< unsigned > data( 182094 );
	for( int i = 0; i < data.size(); i++ )
		data[ i ] = rand() % 531;

	cuda::vector< unsigned > ddata;
	cuda::copy( ddata, data );

	for( int i = 0; i < data.size(); i += segmentSize )
		std::partial_sum( data.data() + i, data.data() + std::min< size_t >( data.size(), i + segmentSize ), data.data() + i );

	cuda::segmented_inclusive_scan( ddata.data(), (int) data.size(), segmentSize, ddata.data() );

	std::vector< unsigned > testdata;
	cuda::copy( testdata, ddata );

	for( int i = 0; i < data.size(); i++ )
		if( data[ i ] != testdata[ i ] )
			printf( "%d: %d, %d\n", i, data[ i ], testdata[ i ] );

	for( int i = 0; i < data.size(); i++ )
		BOOST_REQUIRE( data[ i ] == testdata[ i ] );
}

BOOST_AUTO_TEST_CASE( segmented_exclusive_scan )
{
	int const segmentSize = 1024;

	std::vector< unsigned > data( 109487 );
	for( int i = 0; i < data.size(); i++ )
		data[ i ] = rand() % 491;

	cuda::vector< unsigned > ddata;
	cuda::copy( ddata, data );

	for( int i = 0; i < data.size(); i += segmentSize )
		util::partial_sum_exclusive( data.data() + i, data.data() + std::min< size_t >( data.size(), i + segmentSize ), data.data() + i );

	//cuda::timer t;
	cuda::segmented_exclusive_scan( ddata.data(), (int) data.size(), segmentSize, ddata.data() );

	//float time = t.record_time( "test" );
	//printf( "%.2fms (%.1fGB/s)\n", time, data.capacity() / time * 4000 / 1024 / 1024 / 1024 );

	std::vector< unsigned > testdata;
	cuda::copy( testdata, ddata );

	for( int i = 0; i < data.size(); i++ )
		BOOST_REQUIRE( data[ i ] == testdata[ i ] );
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()