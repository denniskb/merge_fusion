#pragma once

#include <flink/flat_map.h>
#include <flink/math.h>

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

	flink::float4 Minimum() const;
	flink::float4 Maximum() const;

	flink::float4 VoxelCenter( int x, int y, int z ) const;
	flink::float4 ChunkIndex( flink::float4 const & world, int chunkRes ) const;

	flink::flat_map< unsigned, Brick > & Data();
	flink::flat_map< unsigned, Brick > const & Data() const;

private:
	int m_res;
	float m_sideLen;
	float m_truncMargin;

	flink::flat_map< unsigned, Brick > m_data;
};

}