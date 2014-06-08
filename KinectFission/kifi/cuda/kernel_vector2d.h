#pragma once

#include <host_defines.h>



namespace kifi {
namespace cuda {

template< typename T >
class kernel_vector2d
{
public:
	__host__ kernel_vector2d( T * data, unsigned width, unsigned height );

	inline __device__ unsigned width() const;
	inline __device__ unsigned height() const;

	inline __device__ T & operator()( unsigned x, unsigned y );
	inline __device__ T const & operator()( unsigned x, unsigned y ) const;

private:
	T * m_data;
	unsigned m_width;
	unsigned m_height;
};

}} // namespace



#pragma region Implementation



namespace kifi {
namespace cuda {

template< typename T >
inline __host__ kernel_vector2d< T >::kernel_vector2d( T * data, unsigned width, unsigned height ) :
	m_data( data ),
	m_width( width ),
	m_height( height )
{
}



template< typename T >
inline __device__ unsigned kernel_vector2d< T >::width() const
{
	return m_width;
}

template< typename T >
inline __device__ unsigned kernel_vector2d< T >::height() const
{
	return m_height;
}



template< typename T >
inline __device__ T & kernel_vector2d< T >::operator()( unsigned x, unsigned y )
{
	return m_data[ x + y * width() ];
}

template< typename T >
inline __device__ T const & kernel_vector2d< T >::operator()( unsigned x, unsigned y ) const
{
	return m_data[ x + y * width() ];
}

}} // namespace

#pragma endregion