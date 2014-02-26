/*
Voxel volume as used in KinectFusion
*/

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

	/*
	x, y and z map directly to the world coordinate system axes.
	*/
	//Voxel operator()( int x, int y, int z ) const;
	//void operator()( int x, int y, int z, Voxel v );

	std::vector< Voxel > const & Voxels() const;
	std::vector< int > const & VoxelIndices() const;

	//bool operator==( HostVolume const & rhs ) const;
	/*
	Returns true iff all voxels of this and rhs are pairwise 'close'.
	(See Voxel::Close for details.)
	*/
	//bool Close( HostVolume const & rhs, float delta ) const;

	/*
	Returns the world position of the voxel at index (x, y, z)
	*/
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

	std::vector< int > m_voxelIndices;
	std::vector< Voxel > m_voxels;

	int const m_res;
	float const m_sideLen;
	float const m_truncMargin;
	int m_nUpdates;

	//cached values
	float const m_voxelLen;
	float const m_resOver2MinusPoint5TimesVoxelLenNeg;

	static bool IndicesAreValid( int x, int y, int z, int resolution );
	static unsigned Index3Dto1D( unsigned x, unsigned y, unsigned z, unsigned resolution );
};

}