#pragma once

#include <host_defines.h>
#include <vector_functions.h>
#include <vector_types.h>



namespace kifi {
namespace cuda {

class KernelVolume
{
public:
	inline __host__ KernelVolume( unsigned resolution, float sideLength ) :
		m_resolution( resolution ),
		m_sideLength( sideLength )
	{
	}

	inline __device__ float4 ChunkIndex( float4 world, unsigned chunkRes ) const
	{
		return make_float4
		(
			( world.x + m_sideLength * 0.5f ) / m_sideLength * NumChunksInVolume( chunkRes ),
			( world.y + m_sideLength * 0.5f ) / m_sideLength * NumChunksInVolume( chunkRes ),
			( world.z + m_sideLength * 0.5f ) / m_sideLength * NumChunksInVolume( chunkRes ),
			1.0f
		);
	}

	inline __device__ unsigned NumChunksInVolume( unsigned chunkRes ) const
	{
		return m_resolution / chunkRes;
	}

private:
	unsigned m_resolution;
	float m_sideLength;
};

}}