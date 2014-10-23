#pragma once

#include <utility>

#include <kifi/util/math.h>



namespace kifi {

enum KinectDepthSensorResolution
{
	KinectDepthSensorResolution320x240,
	KinectDepthSensorResolution640x480,
};

enum KinectDepthSensorMode
{
	KinectDepthSensorModeNear,
	KinectDepthSensorModeFar
};

class DepthSensorParams
{
public:
	DepthSensorParams
	(
		util::int2   resolutionPixels,
		util::float2 focalLengthNorm,
		util::float2 principalPointPixels,
		util::float2 sensibleDistanceRangeMeters
	);

	util::int2   ResolutionPixels() const;
	util::float2 FocalLengthNorm() const;
	util::float2 PrincipalPointNorm() const;
	util::float2 SensibleRangeMeters() const;

	/*
	Returns a OpenGl-style (z \in [-w, w]) projection matrix which
	projects a point from right-handed eye space to left-handed clip space.
	*/
	util::float4x4 EyeToClipRH() const;

	static DepthSensorParams KinectV1Params( KinectDepthSensorResolution resolution, KinectDepthSensorMode mode );
	static DepthSensorParams KinectV2Params();

private:
	util::int2   m_res;
	util::float2 m_fl;
	util::float2 m_pp;
	util::float2 m_range;
};

} // namespace