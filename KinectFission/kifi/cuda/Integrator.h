#pragma once

#include <kifi/util/DirectXMathExt.h>

#include <kifi/cuda/vector.h>



namespace kifi {
namespace cuda {

class DepthFrame;
class Volume;

class Integrator
{
public:
	void Integrate
	( 
		Volume & volume,
		DepthFrame const & frame,
		unsigned chunkFootPrint,

		util::float4 const & eye,
		util::float4 const & forward,

		util::float4x4 const & viewProjection,
		util::float4x4 const & viewToWorld
	);

private:
	vector< unsigned > m_splattedChunkIndices;

	static void SplatChunks
	(
		Volume const & volume,
		DepthFrame const & frame,
		util::float4x4 const & viewToWorld,
		unsigned chunkFootPrint,

		vector< unsigned > & outChunkIndices
	);
};

}} // namespace