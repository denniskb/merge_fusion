#pragma once

#include <flink/math.h>
#include <flink/vector.h>

#include "Volume.h"



namespace svc {

class DepthFrame;

class Integrator
{
public:
	void Integrate
	( 
		Volume & volume,
		DepthFrame const & frame,
		float truncationMargin,
		int chunkFootPrint,

		flink::float4 const & eye,
		flink::float4 const & forward,

		flink::float4x4 const & viewProjection,
		flink::float4x4 const & viewToWorld
	);

private:
	flink::vector< unsigned > m_splattedChunks;
	flink::vector< char > m_scratchPad;

	static void SplatChunks
	(
		Volume const & volume,
		DepthFrame const & frame,
		flink::float4x4 const & viewToWorld,
		int chunkFootPrint,

		flink::vector< unsigned > & outChunkIndices
	);

	static void ExpandChunks
	( 
		flink::vector< unsigned > & inOutChunkIndices,
		flink::vector< char > & tmpScratchPad
	);

	static void ChunksToBricks
	(
		flink::vector< unsigned > & inOutChunkIndices,
		int chunkFootPrint,

		flink::vector< char > & tmpScratchPad
	);

	static void ExpandChunksHelper
	(
		flink::vector< unsigned > & inOutChunkIndices,
		unsigned delta,
		bool hintDefinitlyDisjunct,

		flink::vector< char > & tmpScratchPad
	);

	static void UpdateVoxels
	(
		Volume & volume,
		DepthFrame const & frame,
		float truncationMargin,

		flink::float4 const & eye,
		flink::float4 const & forward,
		flink::float4x4 const & viewProjection
	);
};

}