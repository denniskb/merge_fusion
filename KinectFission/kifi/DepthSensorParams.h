#pragma once

#include <kifi/util/math.h>



namespace kifi {

enum KinectSensorMode
{
	KinectSensorModeNear,
	KinectSensorModeFar
};

class DepthSensorParams
{
public:
	DepthSensorParams
	(
		float focalLenghtXInPixels,
		float focalLengthYInPixels,

		float principalPointXInPixels,
		float principalPointYInPixels,

		float minimumSensibleDistanceInMeters,
		float maximumSensibleDistanceInMeters
	);

	float FocalLengthXInPixels() const;
	float FocalLengthYInPixels() const;

	float PrincipalPointXInPixels() const;
	float PrincipalPointYInPixels() const;

	float MinimumSensibleDistanceInMeters() const;
	float MaximumSensibleDistanceInMeters() const;

	util::float4x4 ViewToClip() const;

	static DepthSensorParams KinectParams( KinectSensorMode mode );

private:
	float m_flX, m_flY;
	float m_ppX, m_ppY;
	float m_dMin, m_dMax;
};

} // namespace