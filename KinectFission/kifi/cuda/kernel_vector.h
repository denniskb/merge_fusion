#pragma once

#include <host_defines.h>



namespace kifi {
namespace cuda {

template< typename T >
class kernel_vector
{
public:
	__host__ kernel_vector( T * data, unsigned * size );

	__device__ T & operator[]( unsigned x );
	__device__ T const & operator[]( unsigned x ) const;

	__device__ T * push_back_atomic( unsigned n );

	__device__ unsigned const * size() const;

private:
	T * m_data;
	unsigned * m_size;
};

}} // namespace



#pragma region Implementation



namespace kifi {
namespace cuda {

template< typename T >
__host__ kernel_vector< T >::kernel_vector( T * data, unsigned * size ) :
	m_data( data ),
	m_size( size )
{
}



template< typename T >
__device__ T & kernel_vector< T >::operator[]( unsigned x )
{
	return m_data[ x ];
}

template< typename T >
__device__ T const & kernel_vector< T >::operator[]( unsigned x ) const
{
	return m_data[ x ];
}



template< typename T >
__device__ T * kernel_vector< T >::push_back_atomic( unsigned n )
{
	return m_data + atomicAdd( m_size, n );
}

template< typename T >
__device__ unsigned const * kernel_vector< T >::size() const
{
	return m_size;
}

}} // namespace

#pragma endregion