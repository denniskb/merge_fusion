#include "Volume.h"

#include <algorithm>
#include <cassert>
#include <cstdio>

#include <flink/algorithm.h>
#include <flink/math.h>
#include <flink/util.h>

#include "DepthFrame.h"
#include "Voxel.h"



svc::Volume::Volume( int resolution, float sideLength, int footPrint, float truncMargin ) :
	m_res( resolution ),
	m_sideLen( sideLength ),
	m_footPrint( footPrint ),
	m_truncMargin( truncMargin )
{
	assert( resolution > 0 && resolution <= 1024 );
	assert( sideLength > 0.0f );
	assert( footPrint > 0 && footPrint <= resolution );
	assert( truncMargin > 0.0f );

	assert( flink::powerOf2( resolution ) );
	assert( flink::powerOf2( footPrint ) );
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
	return SideLength() / Resolution();
}

float svc::Volume::TruncationMargin() const
{
	return m_truncMargin;
}



int svc::Volume::BrickResolution() const
{
	return m_footPrint;
}

int svc::Volume::BrickSlice() const
{
	return BrickResolution() * BrickResolution();
}

int svc::Volume::BrickVolume() const
{
	return BrickResolution() * BrickSlice();
}

int svc::Volume::NumBricksInVolume() const
{
	return Resolution() / BrickResolution();
}



flink::float4 svc::Volume::Minimum() const
{
	float minimum = -SideLength() * 0.5f;

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
	float maximum = 0.5f * SideLength();

	return flink::float4
	(
		maximum,
		maximum,
		maximum,
		1.0f
	);
}



flink::vector< unsigned > & svc::Volume::Indices()
{
	return m_indices;
}

flink::vector< unsigned > const & svc::Volume::Indices() const
{
	return m_indices;
}

flink::vector< unsigned > & svc::Volume::Voxels()
{
	return m_voxels;
}

flink::vector< unsigned > const & svc::Volume::Voxels() const
{
	return m_voxels;
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

flink::float4 svc::Volume::BrickIndex( flink::float4 const & world ) const
{
	return
		( world - Minimum() ) / ( Maximum() - Minimum() ) *
		flink::make_float4( (float) NumBricksInVolume() );
}