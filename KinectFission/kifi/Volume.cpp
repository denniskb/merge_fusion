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

	m_tmpVoxelLen = SideLength() / Resolution();
	m_tmpVoxelLenOver2PlusMin = m_tmpVoxelLen * 0.5f + Minimum().x;

	m_tmpVoxelLenInv = 1.0f / m_tmpVoxelLen;
	m_tmpNegVoxelLenInvTimesMin = m_tmpVoxelLenInv * -Minimum().x;
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
	float minimum = -m_sideLen * 0.5f;

	return util::float4
	(
		minimum,
		minimum,
		minimum,
		1.0f
	);
}

util::float4 Volume::Maximum() const
{
	float maximum = 0.5f * m_sideLen;

	return util::float4
	(
		maximum,
		maximum,
		maximum,
		1.0f
	);
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