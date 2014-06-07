#include <kifi/cuda/Volume.h>



namespace kifi {
namespace cuda {

Volume::Volume( int resolution, float sideLength, float truncationMargin ) :
	m_res( resolution ),
	m_sideLen( sideLength ),
	m_truncMargin( truncationMargin )
{
}



cuda::KernelVolume Volume::KernelVolume()
{
	return cuda::KernelVolume( m_res, m_sideLen );
}

cuda::KernelVolume const Volume::KernelVolume() const
{
	return cuda::KernelVolume( m_res, m_sideLen );
}

}} // namespace