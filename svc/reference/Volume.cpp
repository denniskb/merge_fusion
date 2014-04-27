#include "Volume.h"

#include <cassert>

#include "Brick.h"
#include "dxmath.h"
#include "flat_map.h"



svc::Volume::Volume
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

	assert( powerOf2( resolution ) );
}



int svc::Volume::Resolution() const
{
	return m_res;
}

float svc::Volume::SideLength() const
{
	return m_sideLen;
}

float svc::Volume::TruncationMargin() const
{
	return m_truncMargin;
}



float svc::Volume::VoxelLength() const
{
	return m_sideLen / m_res;
}

int svc::Volume::NumChunksInVolume( int chunkRes ) const
{
	return m_res / chunkRes;
}

svc::float4 svc::Volume::Minimum() const
{
	float minimum = -m_sideLen * 0.5f;

	return float4
	(
		minimum,
		minimum,
		minimum,
		1.0f
	);
}

svc::float4 svc::Volume::Maximum() const
{
	float maximum = 0.5f * m_sideLen;

	return float4
	(
		maximum,
		maximum,
		maximum,
		1.0f
	);
}



svc::float4 svc::Volume::VoxelCenter( int x, int y, int z ) const
{
	assert( x >= 0 && x < Resolution() );
	assert( y >= 0 && y < Resolution() );
	assert( z >= 0 && z < Resolution() );

	return 
		Minimum() +
		float4
		( 
			( x + 0.5f ) / Resolution(), 
			( y + 0.5f ) / Resolution(), 
			( z + 0.5f ) / Resolution(), 
			1.0f
		) *
		( Maximum() - Minimum() );
}

svc::float4 svc::Volume::ChunkIndex( float4 const & world, int chunkRes ) const
{
	return
		( world - Minimum() ) / ( Maximum() - Minimum() ) *
		make_float4( (float) NumChunksInVolume( chunkRes ) );
}



svc::flat_map< unsigned, svc::Brick > & svc::Volume::Data()
{
	return m_data;
}

svc::flat_map< unsigned, svc::Brick > const & svc::Volume::Data() const
{
	return m_data;
}