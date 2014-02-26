#include <boost/test/auto_unit_test.hpp>

#include <boost/filesystem/operations.hpp>

#pragma warning( push )
#pragma warning( disable: 4100 4267 4515 4996 )
#include <server/DeviceVolume.h>
#pragma warning( pop )

#include <server/DepthStream.h>
#include <server/DeviceDepthFrame.h>
#include <server/flink.h>
#include <server/HostDepthFrame.h>
#include <server/Timer.h>

#include "../test/util.h"



BOOST_AUTO_TEST_SUITE( DeviceVolume )

BOOST_AUTO_TEST_CASE( Integrate )
{
	kppl::DepthStream ds( ( boost::filesystem::current_path() / "../test/content/imrod_v2.depth" ).string().c_str() );

	kppl::HostDepthFrame depth;
	flink::float4x4 view, viewProj, viewToWorld;
	flink::float4 eye, forward;

	ds.NextFrame( depth, view );
	ComputeMatrices( view, eye, forward, viewProj, viewToWorld );
	kppl::DeviceDepthFrame ddepth( depth );

	kppl::DeviceVolume test( 512, 2.0f, 0.04f );

	cudaThreadSynchronize();
	kppl::Timer timer;
	test.Integrate( ddepth, eye, forward, viewProj );
	cudaThreadSynchronize();
	printf( "DeviceVolume::Integrate - %fms\n", timer.Time() * 1000.0 );

	BOOST_REQUIRE( true );
}

BOOST_AUTO_TEST_SUITE_END()