#pragma once



namespace kifi {
namespace util {

template< typename T >
class kahan_sum
{
public:
	kahan_sum();

	kahan_sum & operator+=( T rhs );
	operator T();

private:
	T m_sum;
	T m_c;
};



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

}} // namespace



#pragma region Implementation

#include <functional>
#include <memory>



namespace kifi {
namespace util {

template< typename T >
kahan_sum< T >::kahan_sum() :
	m_sum( T() ),
	m_c  ( T() )
{
}

template< typename T >
kahan_sum< T > & kahan_sum< T >::operator+=( T rhs )
{
	T y   = rhs - m_c;
	T t   = m_sum + y;
	m_c   = t - m_sum - y;
	m_sum = t;

	return * this;
}

template< typename T >
kahan_sum< T >::operator T()
{
	return m_sum;
}



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

}} // namespace

#pragma endregion