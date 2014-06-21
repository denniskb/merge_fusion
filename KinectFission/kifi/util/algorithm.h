#pragma once

#include <cstddef>
#include <iterator>
#include <vector>



namespace kifi {
namespace util {

// Sorting

template< typename T >
void radix_sort
(
	T * first, T * last,
	void * scratchPad
);

template< typename T, typename U >
void radix_sort
(
	T * keys_first, T * keys_last,
	U * values_first,

	void * scratchPad
);

// Set Operations

template
<
	class InputIterator1, class InputIterator2, 
	class InputIterator3, class InputIterator4,
	class OutputIterator1, class OutputIterator2
>
OutputIterator1 set_union
(
	InputIterator1 keys_first1, InputIterator1 keys_last1,
	InputIterator2 keys_first2, InputIterator2 keys_last2,

	InputIterator3 values_first1,
	InputIterator4 values_first2,

	OutputIterator1 keys_result,
	OutputIterator2 values_result
);

}} // namespace



#pragma region Implementation

#include <cstring>
#include <utility>

#include <kifi/util/numeric.h>



namespace kifi {
namespace util {

template< typename T >
void radix_sort
(
	T * first, T * last,
	void * scratchPad
)
{
	using std::swap;

	unsigned cnt[ 256 ];

	T * tmp = reinterpret_cast< T * >( scratchPad );

	for( unsigned shift = 0; shift != 32; shift += 8 )
	{
		std::memset( cnt, 0, sizeof( cnt ) );

		for( T * it = first; it != last; ++it )
			++cnt[ ( * it >> shift ) & 0xff ];

		partial_sum_exclusive( cnt, cnt + 256, cnt );

		for( T * it = first; it != last; ++it )
			tmp[ cnt[ ( * it >> shift ) & 0xff ]++ ] = * it;

		last = tmp + ( last - first );
		swap( first, tmp );
	}
}

template< typename T, typename U >
void radix_sort
(
	T * keys_first, T * keys_last,
	U * values_first,

	void * scratchPad
)
{
	using std::swap;

	unsigned cnt[ 256 ];

	T * keys_tmp = reinterpret_cast< T * >( scratchPad );
	U * values_tmp = reinterpret_cast< U * >( scratchPad ) + ( keys_last - keys_first );

	for( unsigned shift = 0; shift != 32; shift += 8 )
	{
		std::memset( cnt, 0, sizeof( cnt ) );

		for( auto it = keys_first; it != keys_last; ++it )
			++cnt[ ( * it >> shift ) & 0xff ];

		partial_sum_exclusive( cnt, cnt + 256, cnt );

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



template
<
	class InputIterator1, class InputIterator2, 
	class InputIterator3, class InputIterator4,
	class OutputIterator1, class OutputIterator2
>
OutputIterator1 set_union
(
	InputIterator1 keys_first1, InputIterator1 keys_last1,
	InputIterator2 keys_first2, InputIterator2 keys_last2,

	InputIterator3 values_first1,
	InputIterator4 values_first2,

	OutputIterator1 keys_result,
	OutputIterator2 values_result
)
{
	while( keys_first1 != keys_last1 && keys_first2 != keys_last2 )
	{
		int le = ( * keys_first1 <= * keys_first2 );
		int ge = ( * keys_first1 >= * keys_first2 );

		* keys_result++   = le ? * keys_first1   : * keys_first2;
		* values_result++ = le ? * values_first1 : * values_first2;

		std::advance( keys_first1, le );
		std::advance( keys_first2, ge );

		std::advance( values_first1, le );
		std::advance( values_first2, ge );
	}

	while( keys_first1 != keys_last1 )
	{
		* keys_result++ = * keys_first1++;
		* values_result++ = * values_first1++;
	}

	while( keys_first2 != keys_last2 )
	{
		* keys_result++ = * keys_first2++;
		* values_result++ = * values_first2++;
	}

	return keys_result;
}

}} // namespaces

#pragma endregion