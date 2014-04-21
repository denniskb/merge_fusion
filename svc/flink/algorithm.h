#pragma once

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iterator>
#include <utility>
#include <vector>



namespace flink {

template< class RandomAccessIterator >
inline void radix_sort
(
	RandomAccessIterator first, RandomAccessIterator last,
	std::vector< char > & scratchPad
)
{
	typedef std::iterator_traits< RandomAccessIterator > TIter;

	size_t const size = last - first;

	if( size <= 1 )
		return;

	scratchPad.reserve( size * sizeof( TIter::value_type ) );

	radix_sort< TIter::pointer, TIter::pointer >
	( 
		& * first, & * first + size, 
		reinterpret_cast< TIter::pointer >( scratchPad.data() )
	);
}

template< class RandomAccessIterator1, class RandomAccessIterator2 >
inline void radix_sort
(
	RandomAccessIterator1 first, RandomAccessIterator1 last,
	RandomAccessIterator2 tmp
)
{
	using std::swap;

	size_t cnt[ 256 ];

	for( unsigned shift = 0; shift != 32; shift += 8 )
	{
		std::memset( cnt, 0, sizeof( cnt ) );

		for( auto it = first; it != last; ++it )
			++cnt[ ( * it >> shift ) & 0xff ];

		exclusive_scan( cnt, cnt + 256 );

		for( auto it = first; it != last; ++it )
			tmp[ cnt[ ( * it >> shift ) & 0xff ]++ ] = * it;

		last = tmp + ( last - first );
		swap( first, tmp );
	}
}

template< class RandomAccessIterator1, class RandomAccessIterator2 >
inline void radix_sort
(
	RandomAccessIterator1 keys_first, RandomAccessIterator1 keys_last,
	RandomAccessIterator2 values_first,

	std::vector< char > & scratchPad
)
{
	typedef std::iterator_traits< RandomAccessIterator1 > TIterKeys;
	typedef std::iterator_traits< RandomAccessIterator2 > TIterValues;

	size_t const size = keys_last - keys_first;

	if( size <= 1 )
		return;

	scratchPad.reserve( size * ( sizeof( TIterKeys::value_type ) + sizeof( TIterValues::value_type ) ) );

	radix_sort< TIterKeys::pointer, TIterKeys::pointer, TIterValues::pointer, TIterValues::pointer >
	( 
		& * keys_first, & * keys_first + size, 
		reinterpret_cast< TIterKeys::pointer >( scratchPad.data() ),

		& * values_first,
		reinterpret_cast< TIterValues::pointer >( scratchPad.data() + size * sizeof( TIterKeys::value_type ) )
	);
}

template
< 
	class RandomAccessIterator1, class RandomAccessIterator2,
	class RandomAccessIterator3, class RandomAccessIterator4
>
inline void radix_sort
(
	RandomAccessIterator1 keys_first, RandomAccessIterator1 keys_last,
	RandomAccessIterator2 keys_tmp,

	RandomAccessIterator3 values_first,
	RandomAccessIterator4 values_tmp
)
{
	using std::swap;

	size_t cnt[ 256 ];

	for( unsigned shift = 0; shift != 32; shift += 8 )
	{
		std::memset( cnt, 0, sizeof( cnt ) );

		for( auto it = keys_first; it != keys_last; ++it )
			++cnt[ ( * it >> shift ) & 0xff ];

		exclusive_scan( cnt, cnt + 256 );

		for
		( 
			auto it = std::make_pair( keys_first, values_first );
			it.first != keys_last; 
			++it.first, ++it.second
		)
		{
			size_t idst = cnt[ ( * it.first >> shift ) & 0xff ]++;
			  keys_tmp[ idst ] = * it.first;
			values_tmp[ idst ] = * it.second;
		}

		keys_last = keys_tmp + ( keys_last - keys_first );
		swap( keys_first, keys_tmp );
		swap( values_first, values_tmp );
	}
}



template< class OutputIterator >
inline size_t remove_dups( OutputIterator first, OutputIterator last )
{
	auto dst = first;
	for( auto it = first; it != last; ++it )
		if( * it != * dst )
			* ++dst = * it;

	return std::distance( first, dst ) + 1;
}

template< class OutputIterator >
inline void exclusive_scan( OutputIterator first, OutputIterator last )
{
	typedef std::iterator_traits< OutputIterator >::value_type T;

	T acc = T();
	for( ; first != last; ++first )
	{
		T tmp = * first;
		* first = acc;
		acc += tmp;
	}
}



/*
Determines the size of the intersection between the two sets.
@precond [first1, last1) and [first2, last2) are sorted ascendingly and
contain no duplicates.
*/
template< class InputIterator1, class InputIterator2 >
inline size_t intersection_size
(
	InputIterator1 first1, InputIterator1 last1,
	InputIterator2 first2, InputIterator2 last2
)
{
	size_t result = 0;

	while( first1 != last1 && first2 != last2 )
	{
		unsigned lte = ( * first1 <= * first2 );
		unsigned gte = ( * first1 >= * first2 );

		result += lte * gte;

		std::advance( first1, lte );
		std::advance( first2, gte );
	}

	return result;
}

// TODO: Test and document
template< class BidirectionalIterator1, class BidirectionalIterator2, class BidirectionalIterator3 >
inline void merge_unique_backward
(
	BidirectionalIterator1 first1, BidirectionalIterator1 last1,
	BidirectionalIterator2 first2, BidirectionalIterator2 last2,

	BidirectionalIterator3 result_last
)
{
	--first1;
	--first2;

	--last1;
	--last2;

	--result_last;

	while( last1 != first1 && last2 != first2 )
	{
		int gte = -( * last1 >= * last2 );
		int lte = -( * last1 <= * last2 );

		// TODO: Make sure this translates into a cmov
		* result_last-- = gte ? * last1 : * last2;

		std::advance( last1, gte );
		std::advance( last2, lte );
	}

	while( last1 != first1 )
		* result_last-- = * last1--;

	while( last2 != first2 )
		* result_last-- = * last2--;
}

template
<
	class BidirectionalIterator1, class BidirectionalIterator2,
	class BidirectionalIterator3, 
	class BidirectionalIterator5, class BidirectionalIterator6
>
inline void merge_unique_backward
(
	BidirectionalIterator1 keys_first1, BidirectionalIterator1 keys_last1,
	BidirectionalIterator2 values_last1,

	BidirectionalIterator3 keys_first2, BidirectionalIterator3 keys_last2,
	typename std::iterator_traits< BidirectionalIterator2 >::reference value,

	BidirectionalIterator5 keys_result_last,
	BidirectionalIterator6 values_result_last
)
{
	--keys_first1;
	--keys_first2;

	--keys_last1;
	--keys_last2;
	--values_last1;

	--keys_result_last;
	--values_result_last;

	while( keys_last1 != keys_first1 && keys_last2 != keys_first2 )
	{
		int gte = -( * keys_last1 >= * keys_last2 );
		int lte = -( * keys_last1 <= * keys_last2 );

		// TODO: Make sure this translates into a cmov
		* keys_result_last-- = gte ? * keys_last1 : * keys_last2;
		* values_result_last-- = gte ? * values_last1 : value;

		std::advance( keys_last1, gte );
		std::advance( keys_last2, lte );
		std::advance( values_last1, gte );
	}

	while( keys_last1 != keys_first1 )
	{
		* keys_result_last-- = * keys_last1--;
		* values_result_last-- = * values_last1--;
	}

	while( keys_last2 != keys_first2 )
	{
		* keys_result_last-- = * keys_last2--;
		* values_result_last-- = value;
	}
}

}