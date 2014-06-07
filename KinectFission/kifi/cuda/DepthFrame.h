#pragma once

#include <cstddef>

#include <driver_types.h>

#include <kifi/util/vector2d.h>

#include <kifi/cuda/kernel_vector2d.h>
#include <kifi/cuda/vector.h>



namespace kifi {
namespace cuda {

class DepthFrame
{
public:
	DepthFrame();

	unsigned Width() const;
	unsigned Height() const;

	kernel_vector2d< float > KernelFrame();
	kernel_vector2d< const float > KernelFrame() const;

	void CopyFrom( util::vector2d< float > const & frame, cudaStream_t stream = 0 );
	void CopyTo( util::vector2d< float > & frame, cudaStream_t stream = 0 ) const;

private:
	vector< float > m_data;
	size_t m_width;
	size_t m_height;
};

}} // namespace