#pragma once

#include <functional>



namespace kifi {
namespace util {

template< typename T >
struct id : public std::unary_function< T, T >
{
	T const & operator()( T const & x ) const;
};



template< typename T >
struct offset : public std::unary_function< T, T >
{
	offset( T delta );

	T operator()( T const & x ) const;

private:
	T m_delta;
};

}} // namespace



#pragma region Implementation



namespace kifi {
namespace util {

template< typename T >
T const & id< T >::operator()( T const & x ) const
{
	return x;
}



template< typename T >
offset< T >::offset( T delta ) :
	m_delta( delta )
{
}

template< typename T >
T offset< T >::operator()( T const & x ) const
{
	return x + m_delta;
}

}} // namespace

#pragma endregion