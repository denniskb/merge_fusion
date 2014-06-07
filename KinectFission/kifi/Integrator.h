#pragma once

#include <vector>

#include <kifi/util/DirectXMathExt.h>
#include <kifi/util/vector2d.h>



namespace kifi {

class Volume;

class Integrator
{
public:
	void Integrate
	( 
		Volume & volume,
		util::vector2d< float > const & frame,
		int chunkFootPrint,

		util::float4 const & eye,
		util::float4 const & forward,

		util::float4x4 const & viewProjection,
		util::float4x4 const & viewToWorld
	);

private:
	std::vector< unsigned > m_splattedChunks;
	std::vector< char > m_scratchPad;

	static void SplatChunks
	(
		Volume const & volume,
		util::vector2d< float > const & frame,
		util::float4x4 const & viewToWorld,
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
		util::vector2d< float > const & frame,

		util::float4 const & eye,
		util::float4 const & forward,
		util::float4x4 const & viewProjection
	);
};

} // namespace