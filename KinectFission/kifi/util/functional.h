#pragma once



namespace kifi {
namespace util {

template< typename T >
struct id
{
	typedef T argument_type;
	typedef T result_type;

	T const & operator()( T const & x ) const;
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

}} // namespace

#pragma endregion