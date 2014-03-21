#pragma once

#include "vector.h"



namespace svc {

void radix_sort( vector< unsigned > & data );

template< typename T >
void radix_sort( vector< unsigned > & keys, vector< T > & values )
{
	using std::swap;

	unsigned cnt[ 256 ];
	unsigned const mask = 0xff;

	int size = keys.size();
	keys.resize( 2 * size );
	values.resize( 2 * size );

	unsigned * A = keys.begin();
	unsigned * B = keys.begin() + size;
	T * C = values.begin();
	T * D = values.begin() + size;

	for( int shift = 0; shift < 32; shift += 8 )
	{
		std::memset( cnt, 0, 1024 );

		for( int i = 0; i < size; i++ )
			cnt[ ( A[ i ] >> shift ) & mask ]++;

		for( int i = 1; i < 256; i++ )
			cnt[ i ] += cnt[ i - 1 ];

		for( int i = size - 1; i >= 0; i-- )
		{
			int dstIdx = --cnt[ ( A[ i ] >> shift ) & mask ];
			B[ dstIdx ] = A[ i ];
			D[ dstIdx ] = C[ i ];
		}

		swap( A, B );
		swap( C, D );
	}

	keys.resize( size );
	values.resize( size );
}

}