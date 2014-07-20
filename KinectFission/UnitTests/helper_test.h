#pragma once

#include <kifi/DepthSensorParams.h>
#include <kifi/util/math.h>



inline void ComputeMatrices
(
	kifi::util::float4x4 const & worldToView,

	kifi::util::float4 & outEye,
	kifi::util::float4 & outForward,
	kifi::util::float4x4 & outViewProj,
	kifi::util::float4x4 & outViewToWorld
)
{
	using namespace kifi;

	outViewToWorld = util::invert_transform( worldToView );
	outEye = outViewToWorld * util::float4( 0.0f, 0.0f, 0.0f, 1.0f );
	outForward = outViewToWorld * util::float4( 0.0f, 0.0f, -1.0f, 0.0f );
	outViewProj = DepthSensorParams::KinectParams( KinectSensorModeFar ).ViewToClip() * worldToView;
}