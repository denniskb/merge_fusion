#include <cassert>

#include <kifi/DepthSensorParams.h>



namespace kifi {

DepthSensorParams::DepthSensorParams
(
	util::int2   resolutionPixels,
	util::float2 focalLengthPixels,
	util::float2 principalPointPixels,
	util::float2 sensibleDistanceRangeMeters
) :
	m_res  ( resolutionPixels ),
	m_fl   ( focalLengthPixels ),
	m_pp   ( principalPointPixels ),
	m_range( sensibleDistanceRangeMeters )
{
}



util::int2 DepthSensorParams::ResolutionPixels() const
{
	return m_res;
}

util::float2 DepthSensorParams::FocalLengthPixels() const
{
	return m_fl;
}

util::float2 DepthSensorParams::PrincipalPointPixels() const
{
	return m_pp;
}

util::float2 DepthSensorParams::SensibleRangeMeters() const
{
	return m_range;
}



util::float4x4 DepthSensorParams::EyeToClipRH() const
{
	util::float4x4 result;

	result( 0, 0 ) = 2.0f * m_fl.x / m_res.x;
	result( 0, 2 ) = m_res.x - 2.0f * m_pp.x;

	result( 1, 1 ) = 2.0f * m_fl.y / m_res.y;
	result( 1, 2 ) = m_res.y - 2.0f * m_pp.y;

	result( 2, 2 ) = -(m_range.y + m_range.x) / (m_range.y - m_range.x);
	result( 2, 3 ) = -2.0f * m_range.y * m_range.x / (m_range.y - m_range.x);
	
	result( 3, 2 ) = -1.0f;

	return result;
}



// static 
DepthSensorParams DepthSensorParams::KinectParams( KinectDepthSensorResolution resolution, KinectDepthSensorMode mode )
{
	util::int2 res = ( KinectDepthSensorResolution320x240 == resolution ) ? util::int2( 320, 240 ) : util::int2( 640, 480 );
	// TODO: Make sure those are good default values!
	float fl = ( KinectDepthSensorResolution320x240 == resolution ) ? 285.63f : 571.26f;
	util::float2 range = ( KinectDepthSensorModeNear == mode ) ? util::float2( 0.4f, 3.0f ) : util::float2( 0.8f, 4.0f );

	return DepthSensorParams
	(
		res,
		fl,
		util::float2( res.x * 0.5f, res.y * 0.5f ),
		range
	);
}

// static 
DepthSensorParams DepthSensorParams::KinectV2Params()
{
	return DepthSensorParams
	(
		util::int2( 512, 424 ),
		0.870799f * 424,
		//util::float2( 0.50602675f * 512, 0.499133f * 424 ),
		util::float2( 256, 212 ),
		util::float2( 0.5f, 8.0f )
	);
}

} // namespace