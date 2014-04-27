#pragma once

#include <thrust/device_vector.h>

#include <reference/vector2d.h>



namespace svcu {

class DepthFrame
{
public:
	DepthFrame();

	int Width() const;
	int Height() const;

	float * Data();
	float const * Data() const;

	void operator<<( svc::vector2d< float > const & frame );
	void operator>>( svc::vector2d< float > & frame ) const;

private:
	thrust::device_vector< float > m_data;
	int m_width;
	int m_height;
};

}