#include "IntegratorKernels.h"

#include <cassert>

#include "DepthFrame.h"
#include "helper_math_ext.h"
#include "kernel_vector.h"
#include "kernel_vector2d.h"
#include "KernelVolume.h"
#include "vector_functions_ext.h"
#include "vector_types_ext.h"
#include "Volume.h"



#pragma region Kernels

__global__ void _SplatChunksKernel
(
	svcu::KernelVolume const volume,
	// TODO: Test using 2d textures
	svcu::kernel_vector2d< const float > const frame,
	float4x4 viewToWorld,
	unsigned footPrint,

	svcu::kernel_vector< unsigned > outChunkIndices
)
{
	// TODO: Reorder access to improve coherence
	unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned y = threadIdx.y + blockIdx.y * blockDim.y;

	float depth = frame( x, y );

	if( 0.0f == depth )
		return;

	float4 pxView = make_float4
	(
		( x + 0.5f - frame.width() * 0.5f )  / 585.0f * depth,
		( frame.height() * 0.5f - y - 0.5f ) / 585.0f * depth,
		-depth,
		1.0f
	);

	float4 pxWorld = pxView * viewToWorld;
	float4 pxVol = volume.ChunkIndex( pxWorld, footPrint );
	unsigned chunkIndex = packInts
	(
		(unsigned) ( pxVol.x - 0.5f ),
		(unsigned) ( pxVol.y - 0.5f ),
		(unsigned) ( pxVol.z - 0.5f )
	);

	bool chunkIndexValid =
		pxVol.x >= 0.5f &&
		pxVol.y >= 0.5f &&
		pxVol.z >= 0.5f &&
		
		pxVol.x < volume.NumChunksInVolume( footPrint ) - 0.5f &&
		pxVol.y < volume.NumChunksInVolume( footPrint ) - 0.5f &&
		pxVol.z < volume.NumChunksInVolume( footPrint ) - 0.5f;

	unsigned * slot = outChunkIndices.push_back_atomic( chunkIndexValid );
	
	if( chunkIndexValid )
		* slot = chunkIndex;
}

#pragma endregion

#pragma region Wrappers

void svcu::SplatChunksKernel
(
	Volume const & volume,
	DepthFrame const & frame,
	dlh::float4x4 const & viewToWorld,
	unsigned footPrint,

	vector< unsigned > & outChunkIndices
)
{
	assert( 0 == frame.Width() % 16 );
	assert( 0 == frame.Height() % 16 );

	_SplatChunksKernel<<< dim3( frame.Width() / 16, frame.Height() / 16 ), dim3( 16, 16 ) >>>
	(
		volume.KernelVolume(),
		frame.KernelFrame(),
		make_float4x4( viewToWorld ),
		footPrint,
	
		outChunkIndices.kernel_vector()
	);
}

#pragma endregion