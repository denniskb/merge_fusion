#include "DeviceVolume.h"

#include <cassert>

#include "HostVolume.h"



kppl::DeviceVolume::DeviceVolume( HostVolume const & copy )
{
	CopyFrom( copy );
}



kppl::DeviceVolume & kppl::DeviceVolume::operator<<( HostVolume const & rhs )
{
	CopyFrom( rhs );
	return * this;
}

void kppl::DeviceVolume::operator>>( HostVolume & outFrame ) const
{
	assert( outFrame.Resolution() == m_res );

	short * dst = reinterpret_cast< short * >( & outFrame( 0, 0, 0 ) );
	thrust::copy( m_data.cbegin(), m_data.cend(), dst );
}



//KernelDepthFrame KernelObject() const;



void kppl::DeviceVolume::CopyFrom( HostVolume const & copy )
{
	m_res = copy.Resolution();
	m_sideLen = copy.SideLength();
	m_truncMargin = copy.TrunactionMargin();

	int size = copy.Resolution() * copy.Resolution() * copy.Resolution();
	m_data.resize( size );

	short const * src = reinterpret_cast< short const * >( & copy( 0, 0, 0 ) );
	thrust::copy
	( 
		src, 
		src + size,
		m_data.begin()
	);
}