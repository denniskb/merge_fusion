#pragma once

#include <vector>

#include <flink/math.h>



namespace svc {

class DepthFrame;
class Volume;

class Integrator
{
public:
	void Integrate
	( 
		Volume & volume,
		DepthFrame const & frame,
		int chunkFootPrint,

		flink::float4 const & eye,
		flink::float4 const & forward,

		flink::float4x4 const & viewProjection,
		flink::float4x4 const & viewToWorld
	);

private:
	std::vector< unsigned > m_splattedChunks;
	std::vector< char > m_scratchPad;

	static void SplatChunks
	(
		Volume const & volume,
		DepthFrame const & frame,
		flink::float4x4 const & viewToWorld,
		int chunkFootPrint,

		std::vector< unsigned > & outChunkIndices
	);

	static void ExpandChunks
	( 
		std::vector< unsigned > & inOutChunkIndices,
		std::vector< char > & tmpScratchPad
	);

	static void ChunksToBricks
	(
		std::vector< unsigned > & inOutChunkIndices,
		int chunkFootPrint,

		std::vector< char > & tmpScratchPad
	);

	static void ExpandChunksHelper
	(
		std::vector< unsigned > & inOutChunkIndices,
		unsigned delta,
		bool hintDefinitlyDisjunct,

		std::vector< char > & tmpScratchPad
	);

	static void UpdateVoxels
	(
		Volume & volume,
		DepthFrame const & frame,

		flink::float4 const & eye,
		flink::float4 const & forward,
		flink::float4x4 const & viewProjection
	);
};

}