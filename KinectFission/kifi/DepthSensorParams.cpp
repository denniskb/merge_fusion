#include <kifi/DepthSensorParams.h>



namespace kifi {

DepthSensorParams::DepthSensorParams
(
	float focalLenghtXInPixels,
	float focalLengthYInPixels,

	float principalPointXInPixels,
	float principalPointYInPixels,

	float minimumSensibleDistanceInMeters,
	float maximumSensibleDistanceInMeters
) :
	m_flX( focalLenghtXInPixels ),
	m_flY( focalLengthYInPixels ),

	m_ppX( principalPointXInPixels ),
	m_ppY( principalPointYInPixels ),

	m_dMin( minimumSensibleDistanceInMeters ),
	m_dMax( maximumSensibleDistanceInMeters )
{
}



float DepthSensorParams::FocalLengthXInPixels() const
{
	return m_flX;
}

float DepthSensorParams::FocalLengthYInPixels() const
{
	return m_flY;
}



float DepthSensorParams::PrincipalPointXInPixels() const
{
	return m_ppX;
}

float DepthSensorParams::PrincipalPointYInPixels() const
{
	return m_ppY;
}



float DepthSensorParams::MinimumSensibleDistanceInMeters() const
{
	return m_dMin;
}

float DepthSensorParams::MaximumSensibleDistanceInMeters() const
{
	return m_dMax;
}



util::float4x4 DepthSensorParams::ViewToClip() const
{
	return util::perspective_fl_pp_rh
	(
		FocalLengthXInPixels()           , FocalLengthYInPixels(),
		PrincipalPointXInPixels()        , PrincipalPointYInPixels(),
		MinimumSensibleDistanceInMeters(), MaximumSensibleDistanceInMeters()
	);
}



// static 
DepthSensorParams DepthSensorParams::KinectParams( KinectSensorMode mode )
{
	return DepthSensorParams
	(
		585.0f, 585.0f,
		319.5f, 239.5f,
		mode == KinectSensorModeNear ? 0.4f : 0.8f, mode == KinectSensorModeNear ? 3.0f : 4.0f
	);
}

} // namespace