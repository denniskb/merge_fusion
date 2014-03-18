#pragma once

#include "flink.h"
#include "vector.h"



namespace svc {

class HostDepthFrame;
class HostVolume;

class HostIntegrator
{
public:
	void Integrate
	( 
		HostVolume & volume,
		HostDepthFrame const & frame,

		flink::float4 const & eye,
		flink::float4 const & forward,

		flink::float4x4 const & viewProjection,
		flink::float4x4 const & viewToWorld
	);

private:
	vector< unsigned > m_affectedIndices;

	static void SplatBricks
	(
		HostVolume const & volume,
		HostDepthFrame const & depthMap,
		flink::float4x4 const & viewToWorld,

		vector< unsigned > & outBrickIndices
	);

	static void BricksToVoxels
	(
		HostVolume const & volume,
		vector< unsigned > & inOutIndices
	);

	static void UpdateVoxels
	(
		HostVolume & volume,

		svc::HostDepthFrame const & frame, 

		flink::float4 const & eye,
		flink::float4 const & forward,
		flink::float4x4 const & viewProjection
	);
};

}