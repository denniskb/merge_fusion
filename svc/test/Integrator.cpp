#include <boost/test/auto_unit_test.hpp>

#include <cstdio>

#include <boost/filesystem/operations.hpp>

#include <reference/Cache.h>
#include <reference/DepthFrame.h>
#include <reference/Integrator.h>
#include <reference/Volume.h>
#include <reference/DepthStream.h>
#include <reference/flink.h>
#include <reference/util.h>
#include <reference/Voxel.h>

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

	svc::Volume v( 256, 2.0f, 2 );
	svc::Cache cache;

	svc::DepthStream ds( ( boost::filesystem::current_path() / "content/imrod_v2.depth" ).string().c_str() );

	svc::DepthFrame depth;
	flink::float4x4 view, viewProj, viewToWorld;
	flink::float4 eye, forward;

	ds.NextFrame( depth, view );
	ComputeMatrices( view, eye, forward, viewProj, viewToWorld );

	i.Integrate( v, cache, depth, eye, forward, viewProj, viewToWorld );

	FILE * debug;
	fopen_s( & debug, "C:/TEMP/volume_integrate.obj", "w" );

	for( int i = 0; i < v.Indices().size(); i++ )
	{
		svc::Voxel vx = v.Voxels()[ i ];
		if( vx.Weight() == 0 )
			continue;

		unsigned x, y, z;
		svc::unpackInts( v.Indices()[ i ], x, y, z );
		flink::float4 pos = v.VoxelCenter( x, y, z );
		fprintf_s( debug, "v %f %f %f\n", pos.x, pos.y, pos.z );
	}
	
	fclose( debug );

	BOOST_REQUIRE( true );
}

BOOST_AUTO_TEST_SUITE_END()