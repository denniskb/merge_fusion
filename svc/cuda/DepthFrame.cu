#include "DepthFrame.h"

#include <reference/vector2d.h>



svcu::DepthFrame::DepthFrame() :
	m_width( 0 ),
	m_height( 0 )
{
}



int svcu::DepthFrame::Width() const
{
	return m_width;
}

int svcu::DepthFrame::Height() const
{
	return m_height;
}



float * svcu::DepthFrame::Data()
{
	return thrust::raw_pointer_cast( m_data.data() );
}

float const * svcu::DepthFrame::Data() const
{
	return thrust::raw_pointer_cast( m_data.data() );
}



void svcu::DepthFrame::operator<<( svc::vector2d< float > const & frame )
{
	m_data.resize( frame.size() );
	m_width = frame.width();
	m_height = frame.height();

	cudaMemcpy( Data(), & frame( 0, 0 ), frame.size() * sizeof( float ), cudaMemcpyHostToDevice );
}

void svcu::DepthFrame::operator>>( svc::vector2d< float > & frame ) const
{
	frame.resize( Width(), Height() );

	cudaMemcpy( & frame( 0, 0 ), Data(), frame.size() * sizeof( float ), cudaMemcpyDeviceToHost );
}