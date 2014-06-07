#pragma once

#include <functional>



namespace kifi {
namespace util {

template< class InputIterator, class OutputIterator >
OutputIterator partial_sum_exclusive
(
	InputIterator first, InputIterator last, 
	OutputIterator result 
);

template< class InputIterator, class OutputIterator, class BinaryOperation >
OutputIterator partial_sum_exclusive
(
	InputIterator first, InputIterator last, 
	OutputIterator result, BinaryOperation binary_op
);



#pragma region Implementation

template< class InputIterator, class OutputIterator >
OutputIterator partial_sum_exclusive
(
	InputIterator first, InputIterator last, 
	OutputIterator result 
)
{
	typedef std::iterator_traits< InputIterator >::value_type T;

	return partial_sum_exclusive( first, last, result, std::plus< T >() );
}

template< class InputIterator, class OutputIterator, class BinaryOperation >
OutputIterator partial_sum_exclusive
(
	InputIterator first, InputIterator last, 
	OutputIterator result, BinaryOperation binary_op
)
{
	typedef std::iterator_traits< InputIterator >::value_type T;
	
	T acc = T();
	for( ; first != last; ++first, ++result )
	{
		T tmp = * first;
		* result = acc;
		acc = binary_op( acc, tmp );
	}

	return result;
}

#pragma endregion

}} // namespace