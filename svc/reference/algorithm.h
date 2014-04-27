#pragma once

#include <cstddef>
#include <iterator>
#include <vector>



namespace svc {

template< class OutputIterator >
inline void 
exclusive_scan( OutputIterator first, OutputIterator last );

template< class InputIterator1, class InputIterator2 >
inline size_t 
intersection_size
(
	InputIterator1 first1, InputIterator1 last1,
	InputIterator2 first2, InputIterator2 last2
);

template< class BidirectionalIterator1, class BidirectionalIterator2, class BidirectionalIterator3 >
inline void 
set_union_backward
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
inline void 
set_union_backward
(
	BidirectionalIterator1 keys_first1, BidirectionalIterator1 keys_last1,
	BidirectionalIterator2 values_last1,

	BidirectionalIterator3 keys_first2, BidirectionalIterator3 keys_last2,
	typename std::iterator_traits< BidirectionalIterator2 >::reference value,

	BidirectionalIterator4 keys_result_last,
	BidirectionalIterator5 values_result_last
);

template< class RandomAccessIterator >
inline void 
radix_sort
(
	RandomAccessIterator first, RandomAccessIterator last,
	std::vector< char > & scratchPad
);

template< class RandomAccessIterator1, class RandomAccessIterator2 >
inline void 
radix_sort
(
	RandomAccessIterator1 first, RandomAccessIterator1 last,
	RandomAccessIterator2 tmp
);

template< class RandomAccessIterator1, class RandomAccessIterator2 >
inline void
radix_sort
(
	RandomAccessIterator1 keys_first, RandomAccessIterator1 keys_last,
	RandomAccessIterator2 values_first,

	std::vector< char > & scratchPad
);

template
< 
	class RandomAccessIterator1, class RandomAccessIterator2,
	class RandomAccessIterator3, class RandomAccessIterator4
>
inline void
radix_sort
(
	RandomAccessIterator1 keys_first, RandomAccessIterator1 keys_last,
	RandomAccessIterator2 keys_tmp,

	RandomAccessIterator3 values_first,
	RandomAccessIterator4 values_tmp
);

template< class OutputIterator >
inline size_t 
unique( OutputIterator first, OutputIterator last );

}



#include "algorithm.inl"