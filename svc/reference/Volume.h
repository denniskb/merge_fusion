#pragma once

#include "Brick.h"
#include "dxmath.h"
#include "flat_map.h"



namespace svc {

class Volume
{
public:
	Volume( int resolution, float sideLength, float truncationMargin );

	int Resolution() const;
	float SideLength() const;
	float TruncationMargin() const;

	float VoxelLength() const;
	int NumChunksInVolume( int chunkRes ) const;

	float4 Minimum() const;
	float4 Maximum() const;

	float4 VoxelCenter( int x, int y, int z ) const;
	float4 ChunkIndex( float4 const & world, int chunkRes ) const;

	flat_map< unsigned, Brick > & Data();
	flat_map< unsigned, Brick > const & Data() const;

private:
	int m_res;
	float m_sideLen;
	float m_truncMargin;

	flat_map< unsigned, Brick > m_data;
};

}