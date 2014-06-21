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

	float vlen = sideLength / resolution;
	m_tmpVoxelLen = util::set( util::vec3( vlen ), 1.0f );
	m_tmpVoxelLenOver2PlusMin = util::set( util::vec3( vlen * 0.5f + Minimum().x ), 0.0f );

	float vlenInv = 1.0f / vlen;
	m_tmpVoxelLenInv = util::set( util::vec3( vlenInv ), 1.0f );
	m_tmpNegVoxelLenInvTimesMin = m_tmpVoxelLenInv * util::set( util::vec3( -Minimum().x ), 0.0f );
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



util::vec3 Volume::Minimum() const
{
	return util::vec3( -0.5f * m_sideLen );
}

util::vec3 Volume::Maximum() const
{
	return util::vec3( 0.5f * m_sideLen );
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