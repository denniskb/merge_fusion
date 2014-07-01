#pragma once

#include <vector>

#include <kifi/util/math.h>
#include <kifi/util/vector2d.h>



namespace kifi {

class Volume;

class Integrator
{
public:
	void Integrate
	( 
		Volume & volume,
		util::vector2d< float > const & frame,

		util::vec3 eye,
		util::vec3 forward,

		util::matrix const & viewProjection,
		util::matrix4x3 const & viewToWorld
	);

private:
	std::vector< unsigned > m_tmpPointCloud;
	std::vector< unsigned > m_tmpScratchPad;

	static std::size_t DepthMap2PointCloud
	(
		Volume const & volume,
		util::vector2d< float > const & frame,
		util::matrix4x3 const & viewToWorld,

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

		util::vec3 eye,
		util::vec3 forward,
		util::matrix const & viewProjection
	);
};

} // namespace