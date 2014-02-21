#include "DeviceDepthFrame.h"

#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include "HostDepthFrame.h"



kppl::DeviceDepthFrame::DeviceDepthFrame( HostDepthFrame const & copy )
{
	CopyFrom( copy );
}

kppl::DeviceDepthFrame & kppl::DeviceDepthFrame::operator=( HostDepthFrame const & rhs )
{
	CopyFrom( rhs );
	return * this;
}



void kppl::DeviceDepthFrame::CopyTo( HostDepthFrame & outFrame ) const
{
	outFrame.Resize( m_width, m_height );
	thrust::copy( m_data.cbegin(), m_data.cend(), & outFrame( 0, 0 ) );
}



kppl::KernelDepthFrame kppl::DeviceDepthFrame::KernelObject() const
{
	return KernelDepthFrame( thrust::raw_pointer_cast( m_data.data() ), m_width, m_height );
}



void kppl::DeviceDepthFrame::CopyFrom( HostDepthFrame const & copy )
{
	m_width = copy.Width();
	m_height = copy.Height();

	m_data.resize( copy.Resolution() );
	thrust::copy( & copy( 0, 0 ), & copy( 0, 0 ) + copy.Resolution(), m_data.begin() );
}