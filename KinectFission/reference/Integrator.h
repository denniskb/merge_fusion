#pragma once

#include <vector>

#include <dlh/DirectXMathExt.h>
#include <dlh/vector2d.h>



namespace svc {

class Volume;

class Integrator
{
public:
	void Integrate
	( 
		Volume & volume,
		dlh::vector2d< float > const & frame,
		int chunkFootPrint,

		dlh::float4 const & eye,
		dlh::float4 const & forward,

		dlh::float4x4 const & viewProjection,
		dlh::float4x4 const & viewToWorld
	);

private:
	std::vector< unsigned > m_splattedChunks;
	std::vector< char > m_scratchPad;

	static void SplatChunks
	(
		Volume const & volume,
		dlh::vector2d< float > const & frame,
		dlh::float4x4 const & viewToWorld,
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
		dlh::vector2d< float > const & frame,

		dlh::float4 const & eye,
		dlh::float4 const & forward,
		dlh::float4x4 const & viewProjection
	);
};

}