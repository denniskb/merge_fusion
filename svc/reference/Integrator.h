#pragma once

#include <flink/math.h>
#include <flink/vector.h>

#include "Volume.h"



namespace svc {

class DepthFrame;

class Integrator
{
public:
	template< int BrickRes >
	void Integrate
	( 
		Volume< BrickRes > & volume,
		DepthFrame const & frame,

		flink::float4 const & eye,
		flink::float4 const & forward,

		flink::float4x4 const & viewProjection,
		flink::float4x4 const & viewToWorld
	);

private:
	flink::vector< unsigned > m_splattedVoxels;
	flink::vector< char > m_scratchPad;

	template< int BrickRes >
	static void SplatBricks
	(
		Volume< BrickRes > const & volume,
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

	template< int BrickRes >
	static void BricksToVoxels
	(
		Volume< BrickRes > const & volume,
		flink::vector< unsigned > & inOutIndices
	);

	template< int BrickRes >
	static void UpdateVoxels
	(
		Volume< BrickRes > & volume,

		DepthFrame const & frame, 

		flink::float4 const & eye,
		flink::float4 const & forward,
		flink::float4x4 const & viewProjection
	);
};

}