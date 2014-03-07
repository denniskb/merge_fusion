#include <boost/test/auto_unit_test.hpp>

#include <cstdio>
#include <vector>

#include <boost/filesystem/operations.hpp>

#include <server/HostDepthFrame.h>
#include <server/HostVolume.h>
#include <server/DepthStream.h>
#include <server/flink.h>
#include <server/Timer.h>
#include <server/util.h>
#include <server/Voxel.m>

#include "util.h"



BOOST_AUTO_TEST_SUITE( Volume )

BOOST_AUTO_TEST_CASE( Integrate )
{
	/*
	Quick visual test to verify integration works correctly:
	One depth frame is integrated (generated with poly2depth) and then
	all voxels near surface are stored as vertices to an .obj
	*/

	kppl::HostVolume v( 256, 2.0f, 1 );
	kppl::DepthStream ds( ( boost::filesystem::current_path() / "content/imrod_v2.depth" ).string().c_str() );

	kppl::HostDepthFrame depth;
	flink::float4x4 view, viewProj, viewToWorld;
	flink::float4 eye, forward;

	ds.NextFrame( depth, view );
	ComputeMatrices( view, eye, forward, viewProj, viewToWorld );

	v.Integrate( depth, eye, forward, viewProj, viewToWorld );

	FILE * debug;
	fopen_s( & debug, "C:/TEMP/volume_integrate.obj", "w" );

	for( int i = 0; i < v.BrickIndices().size(); i++ )
	{
		kppl::Voxel vx = v.Voxels()[ i ];
		if( vx.Weight() == 0 )
			continue;

		unsigned x, y, z;
		kppl::unpackInts( v.BrickIndices()[ i ], x, y, z );
		flink::float4 pos = v.VoxelCenter( x, y, z );
		fprintf_s( debug, "v %f %f %f\n", pos.x, pos.y, pos.z );
	}
	
	fclose( debug );

	BOOST_REQUIRE( true );
}

BOOST_AUTO_TEST_CASE( Triangulate )
{
	/*
	Quick visual test to verify triangulation works correctly:
	One depth frame is integrated (generated with poly2depth) and then
	the volume is triangulated using mc and stored as an .obj
	*/

	kppl::HostVolume v( 512, 2.0f, 1 );
	kppl::DepthStream ds( ( boost::filesystem::current_path() / "content/imrod_v2.depth" ).string().c_str() );

	kppl::HostDepthFrame depth;
	flink::float4x4 view, viewProj, viewToWorld;
	flink::float4 eye, forward;

	ds.NextFrame( depth, view );
	ComputeMatrices( view, eye, forward, viewProj, viewToWorld );

	v.Integrate( depth, eye, forward, viewProj, viewToWorld );
	kppl::Timer timer;
	v.Triangulate( "C:/TEMP/volume_triangulate.obj" );
	printf( "tri: %fms\n", timer.Time() * 1000.0 );

	BOOST_REQUIRE( true );
}

BOOST_AUTO_TEST_SUITE_END()