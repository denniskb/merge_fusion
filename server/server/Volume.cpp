#include "Volume.h"

#include <cassert>

#include "Voxel.h"



kppl::Volume::Volume( int resolution, float sideLength ) :
	m_res( resolution ),
	m_sideLen( sideLength )
{
	assert( resolution > 0 );
	assert( sideLength > 0.0f );

	m_data.resize( resolution * resolution * resolution );
}



int kppl::Volume::Resolution() const
{
	return m_res;
}



kppl::Voxel const & kppl::Volume::operator()( int x, int y, int z ) const
{
	assert( IndicesAreValid( x, y, z ) );

	return m_data[ Index3Dto1D( x, y, z ) ];
}

kppl::Voxel & kppl::Volume::operator()( int x, int y, int z )
{
	assert( IndicesAreValid( x, y, z ) );

	return m_data[ Index3Dto1D( x, y, z ) ];
}



bool kppl::Volume::IndicesAreValid( int x, int y, int z ) const
{
	return
		x >= 0 &&
		y >= 0 &&
		z >= 0 &&

		x < Resolution() &&
		y < Resolution() &&
		z < Resolution();
}

int kppl::Volume::Index3Dto1D( int x, int y, int z ) const
{
	assert( IndicesAreValid( x, y, z ) );
	assert( Resolution() <= 1024 );

	return ( z * Resolution() + y ) * Resolution() + x;
}