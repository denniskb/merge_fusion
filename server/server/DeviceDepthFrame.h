#pragma once

#include <thrust/device_vector.h>

#include "KernelDepthFrame.h"



namespace kppl {

class HostDepthFrame;

class DeviceDepthFrame
{
public:
	explicit DeviceDepthFrame( HostDepthFrame const & copy );
	DeviceDepthFrame & operator=( HostDepthFrame const & rhs );

	void CopyTo( HostDepthFrame & outFrame ) const;

	KernelDepthFrame KernelObject() const;

private:
	thrust::device_vector< float > m_data;
	int m_width;
	int m_height;

	void CopyFrom( HostDepthFrame const & copy );
};

}