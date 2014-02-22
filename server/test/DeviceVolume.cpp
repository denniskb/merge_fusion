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
#include <server/HostVolume.h>
#include <server/Voxel.m>

#include "util.h"



BOOST_AUTO_TEST_SUITE( DeviceVolume )

BOOST_AUTO_TEST_CASE( assign )
{
	kppl::HostVolume hvol1( 8, 1.0f, 0.01f );
	kppl::DeviceVolume dvol( 8, 1.0f, 0.01f );
	
	kppl::HostVolume hvol2( 8, 1.0f, 0.01f );
	hvol2( 2, 7, 1 ).Update( 0.005f, 0.01f );
	
	dvol << hvol2;
	dvol >> hvol1;
	
	BOOST_REQUIRE( 1 == hvol1( 2, 7, 1 ).Weight() );
	BOOST_REQUIRE_CLOSE( 0.005f, hvol1( 2, 7, 1 ).Distance( 0.01f ), 0.1f );
}

BOOST_AUTO_TEST_CASE( Integrate )
{
	kppl::DepthStream ds( ( boost::filesystem::current_path() / "content/imrod_v2.depth" ).string().c_str() );

	kppl::HostDepthFrame depth;
	flink::float4x4 view, viewProj;
	flink::float4 eye, forward;

	ds.NextFrame( depth, view );
	ComputeMatrices( view, eye, forward, viewProj );
	kppl::DeviceDepthFrame ddepth( depth );

	kppl::DeviceVolume test( 256, 2.0f, 0.04f );
	test.Integrate( ddepth, eye, forward, viewProj );

	kppl::HostVolume reference( 256, 2.0f, 0.04f );
	reference.Integrate( depth, eye, forward, viewProj );

	kppl::HostVolume htest( 256, 2.0f, 0.04f );
	test >> htest;

	// Numerical imprecision due to intrinsics like fmaf, etc.
	BOOST_REQUIRE( htest.Close( reference, 0.0001f ) );
}

BOOST_AUTO_TEST_SUITE_END()