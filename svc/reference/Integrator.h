#pragma once

#include <vector>

#include "dxmath.h"
#include "vector2d.h"



namespace svc {

class Volume;

class Integrator
{
public:
	void Integrate
	( 
		Volume & volume,
		vector2d< float > const & frame,
		int chunkFootPrint,

		float4 const & eye,
		float4 const & forward,

		float4x4 const & viewProjection,
		float4x4 const & viewToWorld
	);

private:
	std::vector< unsigned > m_splattedChunks;
	std::vector< char > m_scratchPad;

	static void SplatChunks
	(
		Volume const & volume,
		vector2d< float > const & frame,
		float4x4 const & viewToWorld,
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
		vector2d< float > const & frame,

		float4 const & eye,
		float4 const & forward,
		float4x4 const & viewProjection
	);
};

}