#pragma once

#include <host_defines.h>
#include <vector_functions.h>
#include <vector_types.h>



namespace svcu {

class KernelVolume
{
public:
	inline __host__ KernelVolume( unsigned resolution, float sideLength ) :
		resolution( resolution ),
		sideLength( sideLength )
	{
	}

	inline __device__ float4 ChunkIndex( float4 world, unsigned chunkRes ) const
	{
		return make_float4
		(
			( world.x + sideLength * 0.5f ) / sideLength * NumChunksInVolume( chunkRes ),
			( world.y + sideLength * 0.5f ) / sideLength * NumChunksInVolume( chunkRes ),
			( world.z + sideLength * 0.5f ) / sideLength * NumChunksInVolume( chunkRes ),
			1.0f
		);
	}

	inline __device__ unsigned NumChunksInVolume( unsigned chunkRes ) const
	{
		return resolution / chunkRes;
	}

private:
	unsigned resolution;
	float sideLength;
};

}