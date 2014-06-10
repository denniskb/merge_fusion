#pragma once

#include <cstddef>
#include <iterator>
#include <vector>



namespace kifi {
namespace util {

// Set Operations

template< class InputIterator1, class InputIterator2 >
size_t intersection_size
(
	InputIterator1 first1, InputIterator1 last1,
	InputIterator2 first2, InputIterator2 last2
);

template< class BidirectionalIterator1, class BidirectionalIterator2, class BidirectionalIterator3 >
void set_union_backward
(
	BidirectionalIterator1 first1, BidirectionalIterator1 last1,
	BidirectionalIterator2 first2, BidirectionalIterator2 last2,

	BidirectionalIterator3 result_last
);

template
<
	class BidirectionalIterator1, class BidirectionalIterator2,
	class BidirectionalIterator3, 
	class BidirectionalIterator4, class BidirectionalIterator5
>
void set_union_backward
(
	BidirectionalIterator1 keys_first1, BidirectionalIterator1 keys_last1,
	BidirectionalIterator2 values_last1,

	BidirectionalIterator3 keys_first2, BidirectionalIterator3 keys_last2,
	typename std::iterator_traits< BidirectionalIterator2 >::reference value,

	BidirectionalIterator4 keys_result_last,
	BidirectionalIterator5 values_result_last
);

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

}} // namespace



#pragma region Implementation

#include <cstring>
#include <utility>

#include <kifi/util/numeric.h>



namespace kifi {
namespace util {

template< class InputIterator1, class InputIterator2 >
size_t intersection_size
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



template< class BidirectionalIterator1, class BidirectionalIterator2, class BidirectionalIterator3 >
void set_union_backward
(
	BidirectionalIterator1 first1, BidirectionalIterator1 last1,
	BidirectionalIterator2 first2, BidirectionalIterator2 last2,

	BidirectionalIterator3 result_last
)
{
	std::reverse_iterator< BidirectionalIterator1 > rfirst1( last1 );
	std::reverse_iterator< BidirectionalIterator1 > rlast1( first1 );

	std::reverse_iterator< BidirectionalIterator2 > rfirst2( last2 );
	std::reverse_iterator< BidirectionalIterator2 > rlast2( first2 );

	std::reverse_iterator< BidirectionalIterator3 > rresult_first( result_last );

	while( rfirst1 != rlast1 && rfirst2 != rlast2 )
	{
		int gte = ( * rfirst1 >= * rfirst2 );
		int lte = ( * rfirst1 <= * rfirst2 );

		// TODO: Make sure this translates into a cmov
		* rresult_first++ = gte ? * rfirst1 : * rfirst2;

		std::advance( rfirst1, gte );
		std::advance( rfirst2, lte );
	}

	while( rfirst1 != rlast1 )
		* rresult_first++ = * rfirst1++;

	while( rfirst2 != rlast2 )
		* rresult_first++ = * rfirst2++;
}



template
<
	class BidirectionalIterator1, class BidirectionalIterator2,
	class BidirectionalIterator3, 
	class BidirectionalIterator4, class BidirectionalIterator5
>
void set_union_backward
(
	BidirectionalIterator1 keys_first1, BidirectionalIterator1 keys_last1,
	BidirectionalIterator2 values_last1,

	BidirectionalIterator3 keys_first2, BidirectionalIterator3 keys_last2,
	typename std::iterator_traits< BidirectionalIterator2 >::reference value,

	BidirectionalIterator4 keys_result_last,
	BidirectionalIterator5 values_result_last
)
{
	std::reverse_iterator< BidirectionalIterator1 > rkeys_first1( keys_last1 );
	std::reverse_iterator< BidirectionalIterator1 > rkeys_last1( keys_first1 );
	std::reverse_iterator< BidirectionalIterator2 > rvalues_first1( values_last1 );

	std::reverse_iterator< BidirectionalIterator3 > rkeys_first2( keys_last2 );
	std::reverse_iterator< BidirectionalIterator3 > rkeys_last2( keys_first2 );

	std::reverse_iterator< BidirectionalIterator4 > rkeys_result_first( keys_result_last );
	std::reverse_iterator< BidirectionalIterator5 > rvalues_result_first( values_result_last );

	while( rkeys_first1 != rkeys_last1 && rkeys_first2 != rkeys_last2 )
	{
		int gte = ( * rkeys_first1 >= * rkeys_first2 );
		int lte = ( * rkeys_first1 <= * rkeys_first2 );

		// TODO: Make sure this translates into a cmov
		* rkeys_result_first++ = gte ? * rkeys_first1 : * rkeys_first2;
		* rvalues_result_first++ = gte ? * rvalues_first1 : value;

		std::advance( rkeys_first1, gte );
		std::advance( rkeys_first2, lte );
		std::advance( rvalues_first1, gte );
	}

	while( rkeys_first1 != rkeys_last1 )
	{
		* rkeys_result_first++ = * rkeys_first1++;
		* rvalues_result_first++ = * rvalues_first1++;
	}

	while( rkeys_first2 != rkeys_last2 )
	{
		* rkeys_result_first++ = * rkeys_first2++;
		* rvalues_result_first++ = value;
	}
}



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

}} // namespaces

#pragma endregion