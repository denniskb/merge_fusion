#include "algorithm.h"

#include <cstring>

#include "vector.h"



void flink::radix_sort( vector< unsigned > & data )
{
	using std::swap;

	unsigned cnt[ 256 ];
	unsigned const mask = 0xff;

	int size = data.size();
	data.resize( 2 * size );

	unsigned * A = data.begin();
	unsigned * B = data.begin() + size;

	for( int shift = 0; shift < 32; shift += 8 )
	{
		std::memset( cnt, 0, 1024 );

		for( int i = 0; i < size; i++ )
			cnt[ ( A[ i ] >> shift ) & mask ]++;

		for( int i = 1; i < 256; i++ )
			cnt[ i ] += cnt[ i - 1 ];

		for( int i = size - 1; i >= 0; i-- )
			B[ --cnt[ ( A[ i ] >> shift ) & mask ] ] = A[ i ];

		swap( A, B );
	}

	data.resize( size );
}



void flink::remove_dups( vector< unsigned > & data )
{
	int idst = 0;
	for( int i = 1; i < data.size(); i++ )
		if( data[ i ] != data[ idst ] )
			data[ ++idst ] = data[ i ];

	data.resize( idst + 1 );
}

void flink::remove_value( vector< unsigned > & data, unsigned value )
{
	int idst = 0;
	for( int i = 0; i < data.size(); i++ )
		if( data[ i ] != value )
			data[ idst++ ] = data[ i ];

	data.resize( idst );
}