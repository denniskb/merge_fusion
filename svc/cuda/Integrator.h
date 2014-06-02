#pragma once

#include <dlh/DirectXMathExt.h>

#include "vector.h"



namespace svcu {

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

		dlh::float4 const & eye,
		dlh::float4 const & forward,

		dlh::float4x4 const & viewProjection,
		dlh::float4x4 const & viewToWorld
	);

private:
	vector< unsigned > m_splattedChunkIndices;

	static void SplatChunks
	(
		Volume const & volume,
		DepthFrame const & frame,
		dlh::float4x4 const & viewToWorld,
		unsigned chunkFootPrint,

		vector< unsigned > & outChunkIndices
	);
};

}