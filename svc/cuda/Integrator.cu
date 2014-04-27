#pragma once

#include <cassert>

#include <vector_functions.h>
#include <vector_types.h>

#include "DepthFrame.h"
#include "helper_math_ext.h"
#include "vector_functions_ext.h"
#include "vector_types_ext.h"
#include "Volume.h"
#include "Integrator.h"
#include "KernelVolume.h"



static __global__ void SplatChunksKernel
(
	svcu::KernelVolume const volume,
	float const * depthFrame, unsigned frameWidth, unsigned frameHeight,
	float4x4 viewToWorld,
	unsigned footPrint,

	unsigned * outChunkIndices,
	unsigned * outChunkIndicesSize
);



svcu::Integrator::Integrator()
{
	m_splattedChunkIndices.resize( 640 * 480 );
	m_splattedChunkIndicesSize.resize( 1 );
}

void svcu::Integrator::Integrate
( 
	Volume & volume,
	DepthFrame const & frame,
	int chunkFootPrint,

	svc::float4 const & eye,
	svc::float4 const & forward,

	svc::float4x4 const & viewProjection,
	svc::float4x4 const & viewToWorld
)
{
	SplatChunks
	( 
		volume, 
		frame, 
		viewToWorld, 
		chunkFootPrint, 

		m_splattedChunkIndices,
		m_splattedChunkIndicesSize
	);
}

// static 
void svcu::Integrator::SplatChunks
(
	Volume const & volume,
	DepthFrame const & frame,
	svc::float4x4 const & viewToWorld,
	int chunkFootPrint,

	thrust::device_vector< unsigned > & outChunkIndices,
	thrust::device_vector< unsigned > & outChunkIndicesSize
)
{
	assert( 0 == frame.Width() % 16 );
	assert( 0 == frame.Height() % 16 );

	outChunkIndices.clear();
	outChunkIndicesSize[ 0 ] = 0;

	cudaEvent_t start, stop;
	cudaEventCreate( & start );
	cudaEventCreate( & stop );

	cudaEventRecord( start );

	SplatChunksKernel<<< dim3( frame.Width() / 16, frame.Height() / 16 ), dim3( 16, 16 ) >>>
	(
		volume.KernelVolume(),
		frame.Data(), frame.Width(), frame.Height(),
		make_float4x4( viewToWorld ),
		chunkFootPrint,

		thrust::raw_pointer_cast( outChunkIndices.data() ),
		thrust::raw_pointer_cast( outChunkIndicesSize.data() )
	);

	cudaEventRecord( stop );
	cudaEventSynchronize( stop );

	float t;
	cudaEventElapsedTime( &t, start, stop );
	printf( "tsplat: %.2fms\n", t );

	outChunkIndices.resize( outChunkIndicesSize[ 0 ] );
}



__global__ void SplatChunksKernel
(
	svcu::KernelVolume const volume,
	float const * depthFrame, unsigned frameWidth, unsigned frameHeight,
	float4x4 viewToWorld,
	unsigned footPrint,

	unsigned * outChunkIndices, unsigned * outChunkIndicesSize
)
{
	unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned y = threadIdx.y + blockIdx.y * blockDim.y;

	float depth = depthFrame[ x + y * frameWidth ];
	
	if( 0.0f == depth )
		return;

	float4 pxView = make_float4
	(
		( x + 0.5f - frameWidth * 0.5f )  / 585.0f * depth,
		( frameHeight * 0.5f - y - 0.5f ) / 585.0f * depth,
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

	unsigned writeMask =
		pxVol.x >= 0.5f &&
		pxVol.y >= 0.5f &&
		pxVol.z >= 0.5f &&
		
		pxVol.x < volume.NumChunksInVolume( footPrint ) - 0.5f &&
		pxVol.y < volume.NumChunksInVolume( footPrint ) - 0.5f &&
		pxVol.z < volume.NumChunksInVolume( footPrint ) - 0.5f;

	unsigned writeOffset = atomicAdd( outChunkIndicesSize, writeMask );

	if( writeMask )
		outChunkIndices[ writeOffset ] = chunkIndex;
}