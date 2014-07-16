#pragma once

#include <fstream>
#include <vector>

#include <kifi/util/math.h>



inline void ComputeMatrices
(
	kifi::util::float4x4 const & view,

	kifi::util::float4 & outEye,
	kifi::util::float4 & outForward,
	kifi::util::float4x4 & outViewProj,
	kifi::util::float4x4 & outViewToWorld
)
{
	using namespace kifi;

	outViewToWorld = util::invert_transform( view );

	outEye = util::float4( 0.0f, 0.0f, 0.0f, 1.0f ) * outViewToWorld;
	outForward = util::float4( 0.0f, 0.0f, -1.0f, 0.0f ) * outViewToWorld;
	outViewProj = view * util::perspective_fov_rh( 0.778633444f, 4.0f / 3.0f, 0.8f, 4.0f );
}