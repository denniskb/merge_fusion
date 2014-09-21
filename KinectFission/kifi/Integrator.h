#pragma once

#include <vector>

#include <kifi/util/math.h>
#include <kifi/util/vector2d.h>

#include <kifi/DepthSensorParams.h>



namespace kifi {

class Volume;

class Integrator
{
public:
	void Integrate
	( 
		Volume & volume,
		util::vector2d< float > const & frame,

		DepthSensorParams const & cameraParams,
		util::float4x4 const & eyeToWorld
	);

private:
	std::vector< unsigned > m_tmpPointCloud;
	std::vector< unsigned > m_tmpScratchPad;

	static std::size_t DepthMap2PointCloud
	(
		Volume const & volume,
		util::vector2d< float > const & frame,

		DepthSensorParams const & cameraParams,
		util::float4x4 const & eyeToWorld,

		std::vector< unsigned > & outPointCloud
	);

	static void ExpandChunks
	( 
		std::vector< unsigned > & inOutChunkIndices,
		std::vector< unsigned > & tmpScratchPad
	);

	static void ExpandChunksHelper
	(
		std::vector< unsigned > & inOutChunkIndices,
		unsigned delta,

		std::vector< unsigned > & tmpScratchPad
	);

	static void UpdateVoxels
	(
		Volume & volume,
		util::vector2d< float > const & frame,
		util::float4x4 const & worldToClip
	);
};

} // namespace