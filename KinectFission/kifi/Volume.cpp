#include "Volume.h"

#include <cassert>

#include <kifi/util/math.h>
#include <kifi/util/flat_map.h>



namespace kifi {

Volume::Volume
( 
	int resolution, 
	float sideLength,
	float truncMargin
) :
	m_res( resolution ),
	m_sideLen( sideLength ),
	m_truncMargin( truncMargin )
{
	assert( resolution > 0 && resolution <= 1024 );
	assert( sideLength > 0.0f );
	assert( truncMargin > 0.0f );

	assert( util::powerOf2( resolution ) );

	m_tmpVoxelLenf = sideLength / resolution;
	m_tmpVoxelLenOver2PlusMinf = m_tmpVoxelLenf * 0.5f + Minimum().x;

	m_tmpVoxelLen = util::set( m_tmpVoxelLenf, m_tmpVoxelLenf, m_tmpVoxelLenf, 1.0f );
	m_tmpVoxelLenOver2PlusMin = util::set( m_tmpVoxelLenOver2PlusMinf, m_tmpVoxelLenOver2PlusMinf, m_tmpVoxelLenOver2PlusMinf, 0.0f );

	float vlenInv = 1.0f / m_tmpVoxelLenf;
	m_tmpVoxelLenInv = util::set( vlenInv, vlenInv, vlenInv, 1.0f );
	m_tmpMinOverNegVoxelLen = m_tmpVoxelLenInv * util::set( -Minimum().x, -Minimum().x, -Minimum().x, 0.0f );
}



int Volume::Resolution() const
{
	return m_res;
}

float Volume::VoxelLength() const
{
	return m_sideLen / m_res;
}

float Volume::SideLength() const
{
	return m_sideLen;
}

float Volume::TruncationMargin() const
{
	return m_truncMargin;
}



util::float4 Volume::Minimum() const
{
	return util::float4( -0.5f * m_sideLen );
}

util::float4 Volume::Maximum() const
{
	return util::float4( 0.5f * m_sideLen );
}



util::flat_map< unsigned, Voxel > & Volume::Data()
{
	return m_data;
}

util::flat_map< unsigned, Voxel > const & Volume::Data() const
{
	return m_data;
}

} // namespace