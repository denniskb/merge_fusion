#include <cstddef>

#include <driver_types.h>

#include <kifi/util/vector2d.h>

#include <kifi/cuda/DepthFrame.h>
#include <kifi/cuda/kernel_vector2d.h>
#include <kifi/cuda/vector.h>



namespace kifi {
namespace cuda {

DepthFrame::DepthFrame() :
	m_width( 0 ),
	m_height( 0 )
{
}



unsigned DepthFrame::Width() const
{
	assert( m_width <= std::numeric_limits< unsigned >::max() );

	return (unsigned) m_width;
}

unsigned DepthFrame::Height() const
{
	assert( m_height <= std::numeric_limits< unsigned >::max() );

	return (unsigned) m_height;
}



kernel_vector2d< float > DepthFrame::KernelFrame()
{
	return kernel_vector2d< float >( m_data.data(), (unsigned) Width(), (unsigned) Height() );
}

kernel_vector2d< const float > DepthFrame::KernelFrame() const
{
	return kernel_vector2d< const float >( m_data.data(), (unsigned) Width(), (unsigned) Height() );
}



void DepthFrame::CopyFrom( util::vector2d< float > const & frame, cudaStream_t stream )
{
	copy( m_data, frame, stream );
	m_width = frame.width();
	m_height = frame.height();
}

void DepthFrame::CopyTo( util::vector2d< float > & frame, cudaStream_t stream ) const
{
	frame.resize( Width(), Height() );
	copy( frame, m_data, stream );
}

}} // namespace