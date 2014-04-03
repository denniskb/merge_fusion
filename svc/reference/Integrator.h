#pragma once

#include <flink/math.h>
#include <flink/vector.h>



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
		DepthFrame const & frame,

		flink::float4 const & eye,
		flink::float4 const & forward,

		flink::float4x4 const & viewProjection,
		flink::float4x4 const & viewToWorld
	);

private:
	flink::vector< unsigned > m_splattedVoxels;
	flink::vector< char > m_scratchPad;

	static void SplatBricks
	(
		Volume const & volume,
		DepthFrame const & frame,
		flink::float4x4 const & viewToWorld,

		flink::vector< unsigned > & outBrickIndices
	);

	static void ExpandBricks
	( 
		flink::vector< unsigned > & inOutBrickIndices,
		flink::vector< char > & tmpScratchPad
	);

	static void ExpandBricksHelper
	(
		flink::vector< unsigned > & inOutBrickIndices,
		unsigned delta,

		flink::vector< char > & tmpScratchPad
	);

	static void BricksToVoxels
	(
		Volume const & volume,
		flink::vector< unsigned > & inOutIndices
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