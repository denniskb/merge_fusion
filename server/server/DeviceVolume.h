#pragma once

#include <thrust/device_vector.h>



namespace kppl {

class HostVolume;

class DeviceVolume
{
public:
	explicit DeviceVolume( HostVolume const & copy );

	DeviceVolume & operator<<( HostVolume const & rhs );
	void operator>>( HostVolume & outFrame ) const;

	//KernelDepthFrame KernelObject() const;

private:
	thrust::device_vector< short > m_data;
	int m_res;
	float m_sideLen;
	int m_truncMargin;

	void CopyFrom( HostVolume const & copy );
};

}