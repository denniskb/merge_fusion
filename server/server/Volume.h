/*
Voxel volume as used in KinectFusion
*/

#pragma once

#include <vector>



namespace kppl {

class Voxel;

class Volume
{
public:
	/*
	Creates a cubic voxel volume with resolution^3 voxels and
	a side length of sideLength meters, centered at the origin.
	*/
	Volume( int resolution, float sideLength );

	int Resolution() const;

	Voxel const & operator()( int x, int y, int z ) const;
	Voxel & operator()( int x, int y, int z );

private:
	int m_res;
	float m_sideLen;

	std::vector< Voxel > m_data;

	bool IndicesAreValid( int x, int y, int z ) const;
	int Index3Dto1D( int x, int y, int z ) const;
};

}