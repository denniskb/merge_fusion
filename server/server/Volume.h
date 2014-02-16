/*
Voxel volume as used in KinectFusion
*/

#pragma once

#include <vector>

#include "flink.h"



namespace kppl {

using namespace flink;

class DepthFrame;
class Voxel;

class Volume
{
public:
	/*
	Creates a cubic voxel volume with resolution^3 voxels and
	a side length of sideLength meters, centered at the origin.
	We use a right-handed coordinate system.
	*/
	Volume( int resolution, float sideLength, float truncationMargin );

	int Resolution() const;

	/*
	x, y and z map directly to the world coordinate system axes.
	*/
	Voxel const & operator()( int x, int y, int z ) const;
	Voxel & operator()( int x, int y, int z );

	/*
	Returns the world position of the voxel at index (x, y, z)
	*/
	float4 VoxelCenter( int x, int y, int z ) const;

	/*
	Integrates a depth frame into the volume using the KinectFusion algorithm.
	@param frame raw depth in mm (0 meaning invalid measurement)
	@param view matrix describing the perspective from which the frame was observed
	@param projection matrix describing the camera lens parameters
	*/
	void Integrate
	(
		DepthFrame const & frame,
		float4x4 const & view,
		float4x4 const & projection
	);

	/*
	Marching cubes ported from http://paulbourke.net/geometry/polygonise/
	@param outOBJ path to a .obj file.
	*/
	void Triangulate( char const * outOBJ );

private:
	int m_res;
	float m_sideLen;
	float m_truncationMargin;

	std::vector< Voxel > m_data;

	bool IndicesAreValid( int x, int y, int z ) const;
	int Index3Dto1D( int x, int y, int z ) const;
};

}