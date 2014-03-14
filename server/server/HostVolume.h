#pragma once

#include "flink.h"
#include "vector.h"



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

	int BrickResolution() const;
	int BrickSlice() const;
	int BrickVolume() const;
	int NumBricksInVolume() const;

	flink::float4 Minimum() const;
	flink::float4 Maximum() const;

	vector< unsigned > & Indices();
	vector< unsigned > & Voxels();

	flink::float4 VoxelCenter( int x, int y, int z ) const;
	flink::float4 BrickIndex( flink::float4 const & world ) const;

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

	int const m_res;
	float const m_sideLen;
	int const m_truncMargin;

	vector< unsigned > m_indices;
	vector< unsigned > m_voxels;

	void MarkBricks
	(
		HostDepthFrame const & depthMap,
		flink::float4x4 const & viewToWorld,

		vector< unsigned > & outBrickIndices
	) const;

	void ExpandBricks( vector< unsigned > & inOutIndices ) const;

	void UpdateVoxels
	(
		vector< unsigned > const & voxelsToUpdate,
		kppl::HostDepthFrame const & frame, 
		flink::float4 const & eye,
		flink::float4 const & forward,
		flink::float4x4 const & viewProjection
	);
};

}