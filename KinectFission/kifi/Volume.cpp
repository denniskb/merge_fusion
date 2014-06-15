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



util::float4 Volume::VoxelCenter( int x, int y, int z ) const
{
	assert( x >= 0 && x < Resolution() );
	assert( y >= 0 && y < Resolution() );
	assert( z >= 0 && z < Resolution() );

	return 
		Minimum() +
		util::float4
		( 
			( x + 0.5f ) / Resolution(), 
			( y + 0.5f ) / Resolution(), 
			( z + 0.5f ) / Resolution(), 
			1.0f
		) *
		( Maximum() - Minimum() );
}

util::float4 Volume::VoxelIndex( util::float4 const & world ) const
{
	return
		( world - Minimum() ) / ( Maximum() - Minimum() ) *
		util::float4( (float) Resolution() );
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