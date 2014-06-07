#include <cassert>

#include <kifi/cuda/DepthFrame.h>
#include <kifi/cuda/IntegratorKernels.h>
#include <kifi/cuda/Volume.h>

#include <kifi/cuda/Integrator.h>

// HACK: 'unreferenced formal parameter' during development
#pragma warning( disable : 4100 )



namespace kifi {
namespace cuda {

void Integrator::Integrate
( 
	Volume & volume,
	DepthFrame const & frame,
	unsigned chunkFootPrint,

	util::float4 const & eye,
	util::float4 const & forward,

	util::float4x4 const & viewProjection,
	util::float4x4 const & viewToWorld
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
void Integrator::SplatChunks
(
	Volume const & volume,
	DepthFrame const & frame,
	util::float4x4 const & viewToWorld,
	unsigned chunkFootPrint,

	vector< unsigned > & outChunkIndices
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

}} // namespace