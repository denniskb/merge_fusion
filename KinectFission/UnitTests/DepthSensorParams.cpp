#include <boost/test/auto_unit_test.hpp>

#include <kifi/DepthSensorParams.h>

#include <kifi/util/math.h>

using namespace kifi;
using namespace kifi::util;



BOOST_AUTO_TEST_SUITE( DepthSensorParamsTest )

BOOST_AUTO_TEST_CASE( ctor )
{
	DepthSensorParams params( int2( 800, 600 ), float2( 0.7f, 0.8f ), float2( 0.49f, 0.51f ), float2( 0.5f, 2.5f ) );

	BOOST_REQUIRE( 800 == params.ResolutionPixels().x() );
	BOOST_REQUIRE( 600 == params.ResolutionPixels().y() );

	BOOST_REQUIRE( 0.7f == params.FocalLengthNorm().x() );
	BOOST_REQUIRE( 0.8f == params.FocalLengthNorm().y() );

	BOOST_REQUIRE( 0.49f == params.PrincipalPointNorm().x() );
	BOOST_REQUIRE( 0.51f == params.PrincipalPointNorm().y() );

	BOOST_REQUIRE( 0.5f == params.SensibleRangeMeters().x() );
	BOOST_REQUIRE( 2.5f == params.SensibleRangeMeters().y() );
}

BOOST_AUTO_TEST_CASE( kinectV1 )
{
	{
		auto params = DepthSensorParams::KinectV1Params( KinectDepthSensorResolution320x240, KinectDepthSensorModeNear );

		BOOST_REQUIRE( 320 == params.ResolutionPixels().x() );
		BOOST_REQUIRE( 240 == params.ResolutionPixels().y() );

		BOOST_REQUIRE( 0.4f == params.SensibleRangeMeters().x() );
		BOOST_REQUIRE( 3.0f == params.SensibleRangeMeters().y() );
	}

	{
		auto params = DepthSensorParams::KinectV1Params( KinectDepthSensorResolution640x480, KinectDepthSensorModeFar );

		BOOST_REQUIRE( 640 == params.ResolutionPixels().x() );
		BOOST_REQUIRE( 480 == params.ResolutionPixels().y() );

		BOOST_REQUIRE( 0.8f == params.SensibleRangeMeters().x() );
		BOOST_REQUIRE( 4.0f == params.SensibleRangeMeters().y() );
	}
}

BOOST_AUTO_TEST_CASE( kinectV2 )
{
	auto params = DepthSensorParams::KinectV2Params();

	BOOST_REQUIRE( 512 == params.ResolutionPixels().x() );
	BOOST_REQUIRE( 424 == params.ResolutionPixels().y() );
}

BOOST_AUTO_TEST_SUITE_END()