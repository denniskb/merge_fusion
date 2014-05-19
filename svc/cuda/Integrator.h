#pragma once

#include <reference/dxmath.h>

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

		svc::float4 const & eye,
		svc::float4 const & forward,

		svc::float4x4 const & viewProjection,
		svc::float4x4 const & viewToWorld
	);

private:
	vector< unsigned > m_splattedChunkIndices;

	static void SplatChunks
	(
		Volume const & volume,
		DepthFrame const & frame,
		svc::float4x4 const & viewToWorld,
		unsigned chunkFootPrint,

		vector< unsigned > & outChunkIndices
	);
};

}