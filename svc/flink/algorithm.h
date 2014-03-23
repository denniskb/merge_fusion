#pragma once

#include "vector.h"



namespace flink {

template< typename T >
inline void radix_sort( vector< T > & data )
{
	using std::swap;

	unsigned cnt[ 256 ];
	unsigned const mask = 0xff;

	int size = data.size();
	data.resize( 2 * size );

	T * A = data.begin();
	T * B = data.begin() + size;

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

template< typename T, typename K >
inline void radix_sort( vector< T > & keys, vector< K > & values )
{
	using std::swap;

	unsigned cnt[ 256 ];
	unsigned const mask = 0xff;

	int size = keys.size();
	keys.resize( 2 * size );
	values.resize( 2 * size );

	T * A = keys.begin();
	T * B = keys.begin() + size;
	K * C = values.begin();
	K * D = values.begin() + size;

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



template< typename T >
inline void remove_dups( vector< T > & data )
{
	int idst = 0;
	for( int i = 1; i < data.size(); i++ )
		if( data[ i ] != data[ idst ] )
			data[ ++idst ] = data[ i ];

	data.resize( idst + 1 );
}

template< typename T >
inline void remove_value( vector< T > & data, T const & value )
{
	int idst = 0;
	for( int i = 0; i < data.size(); i++ )
		if( data[ i ] != value )
			data[ idst++ ] = data[ i ];

	data.resize( idst );
}



/*
Determines the size of the intersection between the two sets.
@precond [first1, last1) and [first2, last2) are sorted ascendingly and
contain no duplicates.
*/
template< typename T >
inline int intersection_size
(
	T const * first1, T const * last1,
	T const * first2, T const * last2
)
{
	int result = 0;

	while( first1 < last1 && first2 < last2 )
	{
		int equal   = ( * first1 == * first2 );
		int less    = ( * first1 <  * first2 );
		int greater = ( * first1 >  * first2 );

		result += equal;
		first1 += equal;
		first2 += equal;

		first1 += less;
		first2 += greater;
	}

	return result;
}

}