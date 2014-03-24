#pragma once

#include <algorithm>

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
	if( 0 == last1 - first1 ||
		0 == last2 - first2 )
		return 0;

	auto tmp = std::lower_bound( first1, last1, * first2 );
	first2 = std::lower_bound( first2, last2, * first1 );
	first1 = tmp;

	int result = 0;
	while( first1 < last1 && first2 < last2 )
	{
		int lte = ( * first1 <= * first2 );
		int gte = ( * first1 >= * first2 );

		result += lte * gte;
		first1 += lte;
		first2 += gte;
	}
	return result;
}

/*
Merges two key-value ranges backwards, which allows
the input and output ranges to overlap

The keys must be sorted ascendingly and be unique.
The merged sequence is unique, too.
*/
template< typename K, typename T >
inline static void merge_unique_backward
(
	K const * const keys_first1, K const * keys_last1,
	T const * values_last1,

	K const * const keys_first2, K const * keys_last2,
	T value, // second range consists of [ value, ..., value )

	K * keys_result_last,
	T * values_result_last
)
{
	assert( keys_first1 <= keys_last1 );
	assert( keys_first2 <= keys_last2 );

	assert( keys_first2 >= keys_last1 || keys_last2 < keys_first1 );

	keys_last1--;
	values_last1--;

	keys_last2--;

	keys_result_last--;
	values_result_last--;

	while( keys_last1 >= keys_first1 && keys_last2 >= keys_first2 )
	{
		int gte = ( * keys_last1 >= * keys_last2 );
		int lte = ( * keys_last1 <= * keys_last2 );

		// TODO: Make sure this translates into a cmov
		* keys_result_last-- = gte ? * keys_last1 : * keys_last2;
		* values_result_last-- = gte ? * values_last1 : value;

		values_last1 -= gte;
		keys_last1 -= gte;
		keys_last2 -= lte;
	}

	while( keys_last1 >= keys_first1 )
	{
		* keys_result_last-- = * keys_last1--;
		* values_result_last-- = * values_last1--;
	}

	while( keys_last2 >= keys_first2 )
	{
		* keys_result_last-- = * keys_last2--;
		* values_result_last-- = value;
	}
}

}