#pragma once

#include <kifi/util/math.h>

#include <kifi/cuda/vector.h>



namespace kifi {
namespace cuda {

class DepthFrame;
class Volume;

void SplatChunksKernel
(
	Volume const & volume,
	DepthFrame const & frame,
	util::float4x4 const & viewToWorld,
	unsigned footPrint,

	vector< unsigned > & outChunkIndices
);

}}