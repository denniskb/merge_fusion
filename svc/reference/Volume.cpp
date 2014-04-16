#include "Volume.h"

#include <cassert>

#include <flink/flat_map.h>
#include <flink/math.h>

#include "Brick.h"



svc::Volume::Volume
( 
	int resolution, 
	float sideLength
) :
	m_res( resolution ),
	m_sideLen( sideLength )
{
	assert( resolution > 0 && resolution <= 1024 );
	assert( sideLength > 0.0f );

	assert( flink::powerOf2( resolution ) );
}



int svc::Volume::Resolution() const
{
	return m_res;
}

float svc::Volume::SideLength() const
{
	return m_sideLen;
}

float svc::Volume::VoxelLength() const
{
	return m_sideLen / m_res;
}

int svc::Volume::NumChunksInVolume( int chunkRes ) const
{
	return m_res / chunkRes;
}

flink::float4 svc::Volume::Minimum() const
{
	float minimum = -m_sideLen * 0.5f;

	return flink::float4
	(
		minimum,
		minimum,
		minimum,
		1.0f
	);
}

flink::float4 svc::Volume::Maximum() const
{
	float maximum = 0.5f * m_sideLen;

	return flink::float4
	(
		maximum,
		maximum,
		maximum,
		1.0f
	);
}



flink::float4 svc::Volume::VoxelCenter( int x, int y, int z ) const
{
	assert( x >= 0 && x < Resolution() );
	assert( y >= 0 && y < Resolution() );
	assert( z >= 0 && z < Resolution() );

	return 
		Minimum() +
		flink::float4
		( 
			( x + 0.5f ) / Resolution(), 
			( y + 0.5f ) / Resolution(), 
			( z + 0.5f ) / Resolution(), 
			1.0f
		) *
		( Maximum() - Minimum() );
}

flink::float4 svc::Volume::ChunkIndex( flink::float4 const & world, int chunkRes ) const
{
	return
		( world - Minimum() ) / ( Maximum() - Minimum() ) *
		flink::make_float4( (float) NumChunksInVolume( chunkRes ) );
}



flink::flat_map< unsigned, svc::Brick > & svc::Volume::Data()
{
	return m_data;
}

flink::flat_map< unsigned, svc::Brick > const & svc::Volume::Data() const
{
	return m_data;
}