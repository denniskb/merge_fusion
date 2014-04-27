#pragma once

#include <thrust/device_vector.h>

#include <reference/dxmath.h>



namespace svcu {

class DepthFrame;
class Volume;

class Integrator
{
public:
	Integrator();

	void Integrate
	( 
		Volume & volume,
		DepthFrame const & frame,
		int chunkFootPrint,

		svc::float4 const & eye,
		svc::float4 const & forward,

		svc::float4x4 const & viewProjection,
		svc::float4x4 const & viewToWorld
	);

private:
	thrust::device_vector< unsigned > m_splattedChunkIndices;
	thrust::device_vector< unsigned > m_splattedChunkIndicesSize;

	static void SplatChunks
	(
		Volume const & volume,
		DepthFrame const & frame,
		svc::float4x4 const & viewToWorld,
		int chunkFootPrint,

		thrust::device_vector< unsigned > & outChunkIndices,
		thrust::device_vector< unsigned > & outChunkIndicesSize
	);
};

}