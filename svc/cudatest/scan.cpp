#include <boost/test/auto_unit_test.hpp>

#include <algorithm>
#include <vector>

#include <dlh/algorithm.h>

#include <cuda/scan.h>
#include <cuda/timer.h>
#include <cuda/vector.h>



BOOST_AUTO_TEST_SUITE( scan )

BOOST_AUTO_TEST_CASE( segmented_inclusive_scan )
{
	int const segmentSize = 256;

	std::vector< unsigned > data( 182094 );
	for( int i = 0; i < data.size(); i++ )
		data[ i ] = rand() % 531;

	svcu::vector< unsigned > ddata;
	svcu::copy( ddata, data );

	for( int i = 0; i < data.size(); i += segmentSize )
		dlh::inclusive_scan( data.data() + i, data.data() + std::min< size_t >( data.size(), i + segmentSize ) );

	svcu::segmented_inclusive_scan( ddata.data(), (int) data.size(), segmentSize, ddata.data() );

	std::vector< unsigned > testdata;
	svcu::copy( testdata, ddata );

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

	svcu::vector< unsigned > ddata;
	svcu::copy( ddata, data );

	for( int i = 0; i < data.size(); i += segmentSize )
		dlh::exclusive_scan( data.data() + i, data.data() + std::min< size_t >( data.size(), i + segmentSize ) );

	//svcu::timer t;
	svcu::segmented_exclusive_scan( ddata.data(), (int) data.size(), segmentSize, ddata.data() );

	//float time = t.record_time( "test" );
	//printf( "%.2fms (%.1fGB/s)\n", time, data.capacity() / time * 4000 / 1024 / 1024 / 1024 );

	std::vector< unsigned > testdata;
	svcu::copy( testdata, ddata );

	for( int i = 0; i < data.size(); i++ )
		BOOST_REQUIRE( data[ i ] == testdata[ i ] );
}

BOOST_AUTO_TEST_SUITE_END()