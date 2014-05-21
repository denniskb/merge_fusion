#include "DepthFrame.h"

#include <cstddef>

#include <driver_types.h>

#include <reference/vector2d.h>

#include "kernel_vector2d.h"
#include "vector.h"



svcu::DepthFrame::DepthFrame() :
	m_width( 0 ),
	m_height( 0 )
{
}



unsigned svcu::DepthFrame::Width() const
{
	assert( m_width <= std::numeric_limits< unsigned >::max() );

	return (unsigned) m_width;
}

unsigned svcu::DepthFrame::Height() const
{
	assert( m_height <= std::numeric_limits< unsigned >::max() );

	return (unsigned) m_height;
}



svcu::kernel_vector2d< float > svcu::DepthFrame::KernelFrame()
{
	return kernel_vector2d< float >( m_data.data(), (unsigned) Width(), (unsigned) Height() );
}

svcu::kernel_vector2d< const float > svcu::DepthFrame::KernelFrame() const
{
	return kernel_vector2d< const float >( m_data.data(), (unsigned) Width(), (unsigned) Height() );
}



void svcu::DepthFrame::CopyFrom( svc::vector2d< float > const & frame, cudaStream_t stream )
{
	copy( m_data, frame, stream );
	m_width = frame.width();
	m_height = frame.height();
}

void svcu::DepthFrame::CopyTo( svc::vector2d< float > & frame, cudaStream_t stream ) const
{
	frame.resize( Width(), Height() );
	copy( frame, m_data, stream );
}