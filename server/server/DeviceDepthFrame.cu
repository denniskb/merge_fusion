#include "DeviceDepthFrame.h"

#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include "HostDepthFrame.h"



kppl::DeviceDepthFrame::DeviceDepthFrame( HostDepthFrame const & copy )
{
	CopyFrom( copy );
}



int kppl::DeviceDepthFrame::Width() const
{
	return m_width;
}

int kppl::DeviceDepthFrame::Height() const
{
	return m_height;
}


	
float * kppl::DeviceDepthFrame::Data()
{
	return thrust::raw_pointer_cast( m_data.data() );
}

float const * kppl::DeviceDepthFrame::Data() const
{
	return thrust::raw_pointer_cast( m_data.data() );
}



kppl::DeviceDepthFrame & kppl::DeviceDepthFrame::operator<<( HostDepthFrame const & rhs )
{
	CopyFrom( rhs );
	return * this;
}

void kppl::DeviceDepthFrame::operator>>( HostDepthFrame & outFrame ) const
{
	outFrame.Resize( m_width, m_height );
	thrust::copy( m_data.cbegin(), m_data.cend(), & outFrame( 0, 0 ) );
}



void kppl::DeviceDepthFrame::CopyFrom( HostDepthFrame const & copy )
{
	m_width = copy.Width();
	m_height = copy.Height();

	m_data.resize( copy.Resolution() );
	
	if( copy.Resolution() > 0 )
		thrust::copy( & copy( 0, 0 ), & copy( 0, 0 ) + copy.Resolution(), m_data.begin() );
}