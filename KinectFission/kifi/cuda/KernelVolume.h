#pragma once

#include <host_defines.h>
#include <vector_types.h>



namespace kifi {
namespace cuda {

class KernelVolume
{
public:
	inline __host__ KernelVolume( unsigned resolution, float sideLength );

	inline __device__ float4 ChunkIndex( float4 world, unsigned chunkRes ) const;
	inline __device__ unsigned NumChunksInVolume( unsigned chunkRes ) const;

private:
	unsigned m_resolution;
	float m_sideLength;
};

}} // namespace



#pragma region Implementation

#include <vector_functions.h>



namespace kifi {
namespace cuda {

__host__ KernelVolume::KernelVolume( unsigned resolution, float sideLength ) :
	m_resolution( resolution ),
	m_sideLength( sideLength )
{
}

__device__ float4 KernelVolume::ChunkIndex( float4 world, unsigned chunkRes ) const
{
	return make_float4
	(
		( world.x + m_sideLength * 0.5f ) / m_sideLength * NumChunksInVolume( chunkRes ),
		( world.y + m_sideLength * 0.5f ) / m_sideLength * NumChunksInVolume( chunkRes ),
		( world.z + m_sideLength * 0.5f ) / m_sideLength * NumChunksInVolume( chunkRes ),
		1.0f
	);
}

__device__ unsigned KernelVolume::NumChunksInVolume( unsigned chunkRes ) const
{
	return m_resolution / chunkRes;
}

}} // namespace

#pragma endregion