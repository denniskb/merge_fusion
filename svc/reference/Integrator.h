#pragma once

#include "flink.h"
#include "vector.h"



namespace svc {

class Cache;
class DepthFrame;
class Volume;

class Integrator
{
public:
	void Integrate
	( 
		Volume & volume,
		Cache & cache,
		DepthFrame const & frame,

		flink::float4 const & eye,
		flink::float4 const & forward,

		flink::float4x4 const & viewProjection,
		flink::float4x4 const & viewToWorld
	);

private:
	vector< unsigned > m_splattedVoxels;

	static void SplatBricks
	(
		Volume const & volume,
		DepthFrame const & frame,
		flink::float4x4 const & viewToWorld,

		vector< unsigned > & outBrickIndices
	);

	static void ExpandBricks
	(
		Volume const & volume,
		Cache & cache,

		vector< unsigned > & inOutBrickIndices
	);

	template< int sliceIdx >
	static void ExpandBricksHelper
	(
		Volume const & volume,
		Cache & cache,
		int deltaLookUp,
		unsigned deltaStore,

		vector< unsigned > & inOutBrickIndices
	);

	static void ExpandBricksHelperX
	(
		int numBricksInVolume,
		vector< unsigned > & inOutBrickIndices
	);

	static void BricksToVoxels
	(
		Volume const & volume,
		vector< unsigned > & inOutIndices
	);

	static void UpdateVoxels
	(
		Volume & volume,

		svc::DepthFrame const & frame, 

		flink::float4 const & eye,
		flink::float4 const & forward,
		flink::float4x4 const & viewProjection
	);
};

}