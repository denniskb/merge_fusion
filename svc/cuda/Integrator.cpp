#include "Integrator.h"

#include <cassert>

#include "DepthFrame.h"
#include "IntegratorKernels.h"
#include "Volume.h"



// HACK: During development
#pragma warning( disable : 4100 )



void svcu::Integrator::Integrate
( 
	Volume & volume,
	DepthFrame const & frame,
	unsigned chunkFootPrint,

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

		m_splattedChunkIndices
	);
}



// static 
void svcu::Integrator::SplatChunks
(
	Volume const & volume,
	DepthFrame const & frame,
	svc::float4x4 const & viewToWorld,
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