/*
Voxel volume as used in KinectFusion
*/

#pragma once

#include <vector>

#include <DirectXMath.h>



namespace kppl {

using namespace DirectX;

class Voxel;

class Volume
{
public:
	/*
	Creates a cubic voxel volume with resolution^3 voxels and
	a side length of sideLength meters, centered at the origin.
	We use a right-handed coordinate system.
	*/
	Volume( int resolution, float sideLength );

	int Resolution() const;

	/*
	x, y and z map directly to the world coordinate system axes.
	*/
	Voxel const & operator()( int x, int y, int z ) const;
	Voxel & operator()( int x, int y, int z );

	/*
	Returns the world position of the voxel at index (x, y, z)
	*/
	XMFLOAT4A VoxelCenter( int x, int y, int z ) const;

	/*
	Integrates a depth frame into the volume using the KinectFusion algorithm.
	@param frame raw depth in mm (0 meaning invalid measurement)
	@param view matrix describing the perspective from which the frame was observed
	@param projection matrix describing the camera lens parameters
	*/
	void Integrate
	(
		std::vector< short > const & frame, 
		XMFLOAT4X4A const & view,
		XMFLOAT4X4A const & projection,
		float truncationMargin
	);

private:
	int m_res;
	float m_sideLen;

	std::vector< Voxel > m_data;

	bool IndicesAreValid( int x, int y, int z ) const;
	int Index3Dto1D( int x, int y, int z ) const;
};

}