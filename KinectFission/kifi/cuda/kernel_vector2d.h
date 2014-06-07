#pragma once

#include <host_defines.h>



namespace kifi {
namespace cuda {

template< typename T >
class kernel_vector2d
{
public:
	inline __host__ kernel_vector2d( T * data, unsigned width, unsigned height ) :
		m_data( data ),
		m_width( width ),
		m_height( height )
	{
	}



	inline __device__ unsigned width() const
	{
		return m_width;
	}

	inline __device__ unsigned height() const
	{
		return m_height;
	}



	inline __device__ T & operator()( unsigned x, unsigned y )
	{
		return m_data[ x + y * width() ];
	}

	inline __device__ T const & operator()( unsigned x, unsigned y ) const
	{
		return m_data[ x + y * width() ];
	}

private:
	T * m_data;
	unsigned m_width;
	unsigned m_height;
};

}} // namespace