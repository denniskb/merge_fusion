#pragma once

#include <reference/dxmath.h>

#include "vector.h"



namespace svcu {

class DepthFrame;
class Volume;

void SplatChunksKernel
(
	Volume const & volume,
	DepthFrame const & frame,
	svc::float4x4 const & viewToWorld,
	unsigned footPrint,

	vector< unsigned > & outChunkIndices
);

}