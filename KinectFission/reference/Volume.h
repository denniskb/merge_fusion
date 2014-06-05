#pragma once

#include <dlh/DirectXMathExt.h>
#include <dlh/flat_map.h>

#include "Brick.h"



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

	dlh::float4 Minimum() const;
	dlh::float4 Maximum() const;

	dlh::float4 VoxelCenter( int x, int y, int z ) const;
	dlh::float4 ChunkIndex( dlh::float4 const & world, int chunkRes ) const;

	dlh::flat_map< unsigned, Brick > & Data();
	dlh::flat_map< unsigned, Brick > const & Data() const;

private:
	int m_res;
	float m_sideLen;
	float m_truncMargin;

	dlh::flat_map< unsigned, Brick > m_data;
};

}