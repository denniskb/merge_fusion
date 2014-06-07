#pragma once

#include <host_defines.h>



namespace kifi {
namespace cuda {

template< typename T >
class kernel_vector
{
public:
	inline __host__ kernel_vector( T * data, unsigned * size ) :
		m_data( data ),
		m_size( size )
	{
	}

	inline __device__ T & operator[]( unsigned x )
	{
		return m_data[ x ];
	}

	inline __device__ T const & operator[]( unsigned x ) const
	{
		return m_data[ x ];
	}

	inline __device__ T * push_back_atomic( unsigned n )
	{
		return m_data + atomicAdd( m_size, n );
	}

	inline __device__ unsigned const * size() const
	{
		return m_size;
	}

private:
	T * m_data;
	unsigned * m_size;
};

}} // namespace