#pragma once

#include <thrust/device_vector.h>

#include "flink.h"



namespace kppl {

class DeviceDepthFrame;
class HostVolume;
class Voxel;

class DeviceVolume
{
public:
	/*
	@precond resolution > 0
	@precond sideLength > 0
	@precond truncationMargin > 0
	*/
	DeviceVolume( int resolution, float sideLength, float truncationMargin );

	int Resolution() const;
	float SideLength() const;
	float TruncationMargin() const;
	
	Voxel * Data();
	Voxel const * Data() const;

	DeviceVolume & operator<<( HostVolume const & rhs );
	void operator>>( HostVolume & outVolume ) const;

	void Integrate
	(
		DeviceDepthFrame const & frame,
		flink::float4 const & eye,
		flink::float4 const & forward,
		flink::float4x4 const & viewProjection
	);

	void Triangulate( char const * outObj ) const;

private:
	// stupid thrust doesn't recognize Voxel as a POD
	thrust::device_vector< unsigned > m_data;
	int m_res;
	float m_sideLen;
	float m_truncMargin;
	int m_nUpdates;
};

}