#include "Volume.h"



svcu::Volume::Volume( int resolution, float sideLength, float truncationMargin ) :
	m_res( resolution ),
	m_sideLen( sideLength ),
	m_truncMargin( truncationMargin )
{
}



svcu::KernelVolume svcu::Volume::KernelVolume()
{
	return svcu::KernelVolume( m_res, m_sideLen );
}

svcu::KernelVolume const svcu::Volume::KernelVolume() const
{
	return svcu::KernelVolume( m_res, m_sideLen );
}