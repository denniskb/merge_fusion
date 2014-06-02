#include "Integrator.h"

#include <cassert>

#include "DepthFrame.h"
#include "IntegratorKernels.h"
#include "Volume.h"

// HACK: 'unreferenced formal parameter' during development
#pragma warning( disable : 4100 )



void svcu::Integrator::Integrate
( 
	Volume & volume,
	DepthFrame const & frame,
	unsigned chunkFootPrint,

	dlh::float4 const & eye,
	dlh::float4 const & forward,

	dlh::float4x4 const & viewProjection,
	dlh::float4x4 const & viewToWorld
)
{
	SplatChunks
	( 
		volume, 
		frame, 
		viewToWorld, 
		chunkFootPrint, 

		m_splattedChunkIndices
	);
}



// static 
void svcu::Integrator::SplatChunks
(
	Volume const & volume,
	DepthFrame const & frame,
	dlh::float4x4 const & viewToWorld,
	unsigned chunkFootPrint,

	svcu::vector< unsigned > & outChunkIndices
)
{
	outChunkIndices.reserve( frame.Width() * frame.Height() );
	outChunkIndices.clear();

	SplatChunksKernel
	(
		volume,
		frame,
		viewToWorld,
		chunkFootPrint,
	
		outChunkIndices
	);
}