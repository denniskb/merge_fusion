#include <boost/test/auto_unit_test.hpp>

#include <cstdio>
#include <vector>

#include <boost/filesystem/operations.hpp>

#include <server/HostDepthFrame.h>
#include <server/HostVolume.h>
#include <server/DepthStream.h>
#include <server/flink.h>
#include <server/Voxel.m>

#include "util.h"



BOOST_AUTO_TEST_SUITE( Volume )

BOOST_AUTO_TEST_CASE( getset )
{
	kppl::HostVolume vol( 10, 1.0f, 0.02f );
	BOOST_REQUIRE( 10 == vol.Resolution() );
	BOOST_REQUIRE_CLOSE( 1.0f, vol.SideLength(), 0.1f );
	BOOST_REQUIRE_CLOSE( 0.02f, vol.TrunactionMargin(), 0.1f );
	BOOST_REQUIRE_CLOSE( 0.1f, vol.VoxelLength(), 0.1f );

	kppl::Voxel v;
	BOOST_REQUIRE( vol( 0, 0, 0 ) == v );
	BOOST_REQUIRE( vol( 3, 1, 9 ) == v );

	v.Update( 1.0f, 1.0f );
	BOOST_REQUIRE( vol( 2, 7, 1 ) != v );

	vol( 1, 1, 1, v );
	BOOST_REQUIRE( vol( 1, 1, 1 ) == v );
}

BOOST_AUTO_TEST_CASE( op_equals )
{
	kppl::HostVolume vol1( 10, 1.0f, 0.02f );
	kppl::HostVolume vol2( 10, 1.0f, 0.02f );
	kppl::HostVolume vol3(  8, 1.0f, 0.02f );

	BOOST_REQUIRE( vol1 == vol2 );
	BOOST_REQUIRE( ! ( vol1 == vol3 ) );

	kppl::Voxel v = vol1( 0, 0, 0 );
	v.Update( 0.02f, 0.02f );
	vol1( 0, 0, 0, v );
	
	BOOST_REQUIRE( ! ( vol1 == vol2 ) );

	v = vol2( 0, 0, 0 );
	v.Update( 0.02f, 0.02f );
	vol2( 0, 0, 0, v );

	BOOST_REQUIRE( vol1 == vol2 );
}

BOOST_AUTO_TEST_CASE( Integrate )
{
	/*
	Quick visual test to verify integration works correctly:
	One depth frame is integrated (generated with poly2depth) and then
	all voxels near surface are stored as vertices to an .obj
	*/

	kppl::HostVolume v( 256, 2.0f, 0.04f );
	kppl::DepthStream ds( ( boost::filesystem::current_path() / "content/imrod_v2.depth" ).string().c_str() );

	kppl::HostDepthFrame depth;
	flink::float4x4 view, viewProj;
	flink::float4 eye, forward;

	ds.NextFrame( depth, view );
	ComputeMatrices( view, eye, forward, viewProj );

	v.Integrate( depth, eye, forward, viewProj );

	FILE * debug;
	fopen_s( & debug, "C:/TEMP/volume_integrate.obj", "w" );

	for( int z = 0; z < v.Resolution(); z++ )
		for( int y = 0; y < v.Resolution(); y++ )
			for( int x = 0; x < v.Resolution(); x++ )
			{
				kppl::Voxel vx = v( x, y, z );
				if( vx.Weight() > 0 && abs( vx.Distance( 0.02f ) ) < 0.005f )
				{
					flink::float4 pos = v.VoxelCenter( x, y, z );
					fprintf_s( debug, "v %f %f %f\n", pos.x, pos.y, pos.z );
				}
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

	kppl::HostVolume v( 256, 2.0f, 0.04f );
	kppl::DepthStream ds( ( boost::filesystem::current_path() / "content/imrod_v2.depth" ).string().c_str() );

	kppl::HostDepthFrame depth;
	flink::float4x4 view, viewProj;
	flink::float4 eye, forward;

	ds.NextFrame( depth, view );
	ComputeMatrices( view, eye, forward, viewProj );

	v.Integrate( depth, eye, forward, viewProj );
	v.Triangulate( "C:/TEMP/volume_triangulate.obj" );

	BOOST_REQUIRE( true );
}

BOOST_AUTO_TEST_SUITE_END()