#pragma once

#include <vector>

#include "flink.h"



namespace kppl {

class HostDepthFrame;
class Voxel;

class HostVolume
{
public:
	/*
	Creates a cubic voxel volume with resolution^3 voxels and
	a side length of sideLength meters, centered at the origin.
	
	'truncationMargin' is in voxels.
	
	@precond resolution \in [1, 1024] and power of 2
	@precond sideLength > 0
	@precond truncationMargin \in [1, resolution] and power of 2
	*/
	HostVolume( int resolution, float sideLength, int truncationMargin );

	int Resolution() const;
	float SideLength() const;
	float VoxelLength() const;
	float TruncationMargin() const; // in meters

	flink::float4 Minimum() const;
	flink::float4 Maximum() const;

	std::vector< unsigned > const & BrickIndices() const;
	std::vector< unsigned > const & Voxels() const;

	flink::float4 VoxelCenter( int x, int y, int z ) const;
	flink::float4 BrickIndex( flink::float4 const & world ) const;

	/*
	Integrates a depth frame into the volume using the KinectFusion algorithm.
	*/
	void Integrate
	(
		HostDepthFrame const & frame,
		flink::float4 const & eye,
		flink::float4 const & forward,
		flink::float4x4 const & viewProjection,
		flink::float4x4 const & viewToWorld
	);

	/*
	Marching cubes ported from http://paulbourke.net/geometry/polygonise/
	@param outOBJ path to a .obj file.
	*/
	void Triangulate( char const * outOBJ ) const;

	/*
	LUT used for Triangulate
	*/
	static int const * TriTable();

private:
	HostVolume & operator=( HostVolume const & rhs );

	std::vector< unsigned > m_brickIndices;
	std::vector< unsigned > m_voxels;

	mutable std::vector< unsigned > m_scratchPad;

	int const m_res;
	float const m_sideLen;
	int const m_truncMargin;
	int m_nUpdates;
};

}