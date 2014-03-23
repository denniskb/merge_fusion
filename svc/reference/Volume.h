#pragma once

#include <flink/math.h>
#include <flink/vector.h>



namespace svc {

class DepthFrame;
class Voxel;

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
	Volume( int resolution, float sideLength, int footPrint, float truncationMargin );

	int Resolution() const;
	float SideLength() const;
	float VoxelLength() const;
	float TruncationMargin() const; // in meters

	int BrickResolution() const;
	int BrickSlice() const;
	int BrickVolume() const;
	int NumBricksInVolume() const;

	flink::float4 Minimum() const;
	flink::float4 Maximum() const;

	flink::vector< unsigned > & Indices();
	flink::vector< unsigned > const & Indices() const;
	flink::vector< unsigned > & Voxels();
	flink::vector< unsigned > const & Voxels() const;

	flink::float4 VoxelCenter( int x, int y, int z ) const;
	flink::float4 BrickIndex( flink::float4 const & world ) const;

private:
	Volume & operator=( Volume const & rhs );

	int const m_res;
	float const m_sideLen;
	int const m_footPrint;
	float const m_truncMargin;

	flink::vector< unsigned > m_indices;
	flink::vector< unsigned > m_voxels;
};

}