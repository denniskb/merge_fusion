#include <boost/test/auto_unit_test.hpp>

#include <algorithm>
#include <vector>

#include <dlh/algorithm.h>

#include <cuda/reduce.h>
#include <cuda/timer.h>
#include <cuda/vector.h>



BOOST_AUTO_TEST_SUITE( reduce )

BOOST_AUTO_TEST_CASE( segmented_reduce )
{
	int const segmentSize = 256;

	std::vector< unsigned > data( 182742 );
	std::vector< unsigned > sums( ( data.size() + segmentSize - 1 ) / segmentSize );
	std::vector< unsigned > testSums;

	for( int i = 0; i < data.size(); i++ )
		data[ i ] = rand() % 319;

	for( int i = 0; i < data.size(); i += segmentSize )
		sums[ i / segmentSize ] = 
			dlh::reduce( data.data() + i, data.data() + std::min< size_t >( data.size(), i + segmentSize ) );

	svcu::vector< unsigned > ddata;
	svcu::vector< unsigned > dsums;
	dsums.resize( sums.size() );

	svcu::copy( ddata, data );

	//svcu::timer t;
	svcu::segmented_reduce( ddata.data(), (int) data.size(), segmentSize, dsums.data() );

	//float time = t.record_time( "test" );
	//printf( "%.2fms (%.1fGB/s)\n", time, data.capacity() / time * 4000 / 1024 / 1024 / 1024 );

	svcu::copy( testSums, dsums );
	
	for( int i = 0; i < sums.size(); i++ )
		BOOST_REQUIRE( sums[ i ] == testSums[ i ] );
}

BOOST_AUTO_TEST_SUITE_END()