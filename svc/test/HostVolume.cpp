#include <boost/test/auto_unit_test.hpp>

#include <cstdio>

#include <boost/filesystem/operations.hpp>

#include <reference/HostDepthFrame.h>
#include <reference/HostIntegrator.h>
#include <reference/HostVolume.h>
#include <reference/DepthStream.h>
#include <reference/flink.h>
#include <reference/util.h>
#include <reference/Voxel.h>

#include "util.h"



BOOST_AUTO_TEST_SUITE( Volume )

BOOST_AUTO_TEST_CASE( ctor )
{
	svc::HostVolume v( 128, 2.0f, 2 );

	BOOST_REQUIRE( v.Resolution() == 128 );
	BOOST_REQUIRE( v.SideLength() == 2.0f );
	BOOST_REQUIRE( v.VoxelLength() == 2.0f / 128.0f );
	BOOST_REQUIRE( v.TruncationMargin() == 2.0f / 128.0f * 2 );

	BOOST_REQUIRE( v.BrickResolution() == 2 );
	BOOST_REQUIRE( v.BrickSlice() == 4 );
	BOOST_REQUIRE( v.BrickVolume() == 8 );
	BOOST_REQUIRE( v.NumBricksInVolume() == 64 );

	BOOST_REQUIRE( v.Minimum().x == -1.0f );
	BOOST_REQUIRE( v.Maximum().y == 1.0f );
}

BOOST_AUTO_TEST_CASE( Triangulate )
{
	/*
	Quick visual test to verify triangulation works correctly:
	One depth frame is integrated (generated with poly2depth) and then
	the volume is triangulated using mc and stored as an .obj
	*/
	svc::HostIntegrator i;

	svc::HostVolume v( 256, 2.0f, 1 );
	svc::DepthStream ds( ( boost::filesystem::current_path() / "content/imrod_v2.depth" ).string().c_str() );

	svc::HostDepthFrame depth;
	flink::float4x4 view, viewProj, viewToWorld;
	flink::float4 eye, forward;

	ds.NextFrame( depth, view );
	ComputeMatrices( view, eye, forward, viewProj, viewToWorld );

	i.Integrate( v, depth, eye, forward, viewProj, viewToWorld );
	v.Triangulate( "C:/TEMP/volume_triangulate.obj" );

	BOOST_REQUIRE( true );
}

BOOST_AUTO_TEST_SUITE_END()