#pragma once

#include <kifi/util/DirectXMathExt.h>
#include <kifi/util/flat_map.h>

#include <kifi/Brick.h>



namespace kifi {

class Volume
{
public:
	Volume( int resolution, float sideLength, float truncationMargin );

	int Resolution() const;
	float SideLength() const;
	float TruncationMargin() const;

	float VoxelLength() const;
	int NumChunksInVolume( int chunkRes ) const;

	util::float4 Minimum() const;
	util::float4 Maximum() const;

	util::float4 VoxelCenter( int x, int y, int z ) const;
	util::float4 ChunkIndex( util::float4 const & world, int chunkRes ) const;

	util::flat_map< unsigned, Brick > & Data();
	util::flat_map< unsigned, Brick > const & Data() const;

private:
	int m_res;
	float m_sideLen;
	float m_truncMargin;

	util::flat_map< unsigned, Brick > m_data;
};

}