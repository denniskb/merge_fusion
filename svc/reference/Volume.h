#pragma once

#include <flink/flat_map.h>
#include <flink/math.h>

#include "Brick.h"



namespace svc {

class DepthFrame;

template< int BrickRes >
class Volume
{
public:
	/*
	Creates a cubic voxel volume with resolution^3 voxels and
	a side length of sideLength meters, centered at the origin.
	
	'truncationMargin' is in voxels.
	
	@precond resolution \in [1, 1024] and power of 2
	@precond sideLength > 0
	@precond footPrint foot print in voxels of depth sample, must be power of 2
	*/
	Volume( int resolution, float sideLength, float truncationMargin );

	int Resolution() const;
	float SideLength() const;
	float TruncationMargin() const; // in meters

	float VoxelLength() const;
	int BrickSlice() const;
	int BrickVolume() const;
	int NumBricksInVolume() const;

	flink::float4 Minimum() const;
	flink::float4 Maximum() const;

	flink::float4 VoxelCenter( int x, int y, int z ) const;
	flink::float4 BrickIndex( flink::float4 const & world ) const;

	flink::flat_map< unsigned, Brick< BrickRes > > & Data();
	flink::flat_map< unsigned, Brick< BrickRes > > const & Data() const;

private:
	int m_res;
	float m_sideLen;
	float m_truncMargin;

	flink::flat_map< unsigned, Brick< BrickRes > > m_data;
};

}