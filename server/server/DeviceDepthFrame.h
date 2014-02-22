#pragma once

#include <thrust/device_vector.h>



namespace kppl {

class HostDepthFrame;

class DeviceDepthFrame
{
public:
	explicit DeviceDepthFrame( HostDepthFrame const & copy );

	int Width() const;
	int Height() const;
	
	float * Data();
	float const * Data() const;

	DeviceDepthFrame & operator<<( HostDepthFrame const & rhs );
	void operator>>( HostDepthFrame & outFrame ) const;

private:
	thrust::device_vector< float > m_data;
	int m_width;
	int m_height;

	void CopyFrom( HostDepthFrame const & copy );
};

}