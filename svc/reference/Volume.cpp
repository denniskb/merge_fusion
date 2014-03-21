#include "Volume.h"

#include <algorithm>
#include <cassert>
#include <cstdio>

#include "flink.h"
#include "DepthFrame.h"
#include "radix_sort.h"
#include "util.h"
#include "Voxel.h"



svc::Volume::Volume( int resolution, float sideLength, int truncationMargin ) :
	m_res( resolution ),
	m_sideLen( sideLength ),
	m_truncMargin( truncationMargin )
{
	assert( resolution > 0 && resolution <= 1024 );
	assert( sideLength > 0.0f );
	assert( truncationMargin > 0 && truncationMargin <= resolution );

	assert( powerOf2( resolution ) );
	assert( powerOf2( truncationMargin ) );
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
	return m_truncMargin * VoxelLength();
}



int svc::Volume::BrickResolution() const
{
	return m_truncMargin;
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
	return Resolution() / m_truncMargin;
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



svc::vector< unsigned > & svc::Volume::Indices()
{
	return m_indices;
}

svc::vector< unsigned > const & svc::Volume::Indices() const
{
	return m_indices;
}

svc::vector< unsigned > & svc::Volume::Voxels()
{
	return m_voxels;
}

svc::vector< unsigned > const & svc::Volume::Voxels() const
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