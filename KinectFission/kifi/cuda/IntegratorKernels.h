#pragma once

#include <dlh/DirectXMathExt.h>

#include "vector.h"



namespace svcu {

class DepthFrame;
class Volume;

void SplatChunksKernel
(
	Volume const & volume,
	DepthFrame const & frame,
	dlh::float4x4 const & viewToWorld,
	unsigned footPrint,

	vector< unsigned > & outChunkIndices
);

}