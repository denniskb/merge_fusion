#pragma once

#include "force_nvcc.h"

#include "DeviceDepthFrame.h"



namespace kppl {

class KernelDepthFrame
{
public:
	inline __host__ KernelDepthFrame( DeviceDepthFrame const & copy ) :
		m_data( copy.Data() ), 
		width( copy.Width() ), 
		height( copy.Height() )
	{
	}

	inline __device__ float operator()( int x, int y ) const
	{
		return m_data[ x + y * width ];
	}

	int const width;
	int const height;

private:
	float const * m_data;
};

}