#pragma once

#include <kifi/cuda/KernelVolume.h>



namespace kifi {
namespace cuda {

class Volume
{
public:
	Volume( int resolution, float sideLength, float truncationMargin );

	cuda::KernelVolume KernelVolume();
	cuda::KernelVolume const KernelVolume() const;

private:
	int m_res;
	float m_sideLen;
	float m_truncMargin;
};

}} // namespace