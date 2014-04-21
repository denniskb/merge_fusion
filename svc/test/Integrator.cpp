#include <boost/test/auto_unit_test.hpp>

#include <cstdio>

#include <boost/filesystem/operations.hpp>

#include <flink/math.h>
#include <flink/util.h>

#include <reference/DepthFrame.h>
#include <reference/DepthStream.h>
#include <reference/Integrator.h>
#include <reference/Volume.h>

#include "util.h"



BOOST_AUTO_TEST_SUITE( Integrator )

BOOST_AUTO_TEST_CASE( Integrate )
{
	/*
	Quick visual test to verify integration works correctly:
	One depth frame is integrated (generated with poly2depth) and then
	all voxels near surface are stored as vertices to an .obj
	*/
	svc::Integrator i;

	svc::Volume v( 512, 2.0f, 0.02f );

	svc::DepthStream ds( ( boost::filesystem::current_path() / "content/imrod_v2.depth" ).string().c_str() );

	svc::DepthFrame depth;
	flink::float4x4 view, viewProj, viewToWorld;
	flink::float4 eye, forward;

	ds.NextFrame( depth, view );
	ComputeMatrices( view, eye, forward, viewProj, viewToWorld );

	i.Integrate( v, depth, 2, eye, forward, viewProj, viewToWorld );
	i.Integrate( v, depth, 2, eye, forward, viewProj, viewToWorld );

	FILE * debug;
	fopen_s( & debug, "C:/TEMP/volume_integrate.obj", "w" );

	// TODO: Adapt code to work with bricks
	for( auto it = v.Data().keys_cbegin(), end = v.Data().keys_cend(); it != end; ++it )
	{
		unsigned x, y, z;
		flink::unpackInts( * it, x, y, z );
		
		flink::float4 pos = v.VoxelCenter( x, y, z );
		
		fprintf_s( debug, "v %f %f %f\n", pos.x, pos.y, pos.z );
	}
	
	fclose( debug );

	BOOST_REQUIRE( true );
}

BOOST_AUTO_TEST_SUITE_END()