#include <boost/test/auto_unit_test.hpp>

#include <algorithm>
#include <vector>

#include <reference/algorithm.h>

#include <cuda/reduce.h>
#include <cuda/timer.h>
#include <cuda/vector.h>



BOOST_AUTO_TEST_SUITE( reduce )

BOOST_AUTO_TEST_CASE( segmented_reduce )
{
	int const segmentSize = 512;

	std::vector< unsigned > data( 182742 );
	std::vector< unsigned > sums( ( data.size() + segmentSize - 1 ) / segmentSize );
	std::vector< unsigned > testSums;

	for( int i = 0; i < data.size(); i++ )
		data[ i ] = rand() % 319;

	for( int i = 0; i < data.size(); i += segmentSize )
		sums[ i / segmentSize ] = 
			svc::reduce( data.data() + i, data.data() + std::min< size_t >( data.size(), i + segmentSize ) );

	svcu::vector< unsigned > ddata;
	svcu::vector< unsigned > dsums;
	dsums.resize( sums.size() );

	svcu::copy( ddata, data );

	svcu::segmented_reduce( ddata.data(), (int) data.size(), segmentSize, dsums.data() );
	
	svcu::copy( testSums, dsums );
	
	for( int i = 0; i < sums.size(); i++ )
		BOOST_REQUIRE( sums[ i ] == testSums[ i ] );
}

BOOST_AUTO_TEST_SUITE_END()