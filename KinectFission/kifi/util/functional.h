#pragma once



namespace kifi {
namespace util {

template< typename T >
struct id
{
	typedef T argument_type;
	typedef T result_type;

	T operator()( T const & x ) const;
};

}} // namespace



#pragma region Implementation



namespace kifi {
namespace util {

template< typename T >
T id< T >::operator()( T const & x ) const
{
	return x;
}

}} // namespace

#pragma endregion