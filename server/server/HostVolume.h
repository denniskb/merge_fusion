/*
Voxel volume as used in KinectFusion
*/

#pragma once

#include "vector.h"

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
	We use a right-handed coordinate system.
	@precond resolution > 0
	@precond sideLength > 0
	@precond truncationMargin > 0
	*/
	HostVolume( int resolution, float sideLength, float truncationMargin );

	int Resolution() const;
	float SideLength() const;
	float VoxelLength() const;
	float TrunactionMargin() const;

	kppl::vector< unsigned > const & VoxelIndices() const;
	kppl::vector< unsigned > const & Voxels() const;

	flink::float4 VoxelCenter( int x, int y, int z ) const;

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
	//void Triangulate( char const * outOBJ ) const;

	/*
	LUT used for Triangulate
	*/
	static int const * TriTable();

private:
	HostVolume & operator=( HostVolume const & rhs );

	kppl::vector< unsigned > m_voxelIndices;
	kppl::vector< unsigned > m_voxels;

	int const m_res;
	float const m_sideLen;
	float const m_truncMargin;
	int m_nUpdates;

	//cached values
	float const m_voxelLen;
	float const m_resOver2MinusPoint5TimesVoxelLenNeg;
};

}