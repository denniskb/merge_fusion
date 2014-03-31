#include <boost/test/auto_unit_test.hpp>

#include <boost/filesystem/operations.hpp>

#include <flink/math.h>

#include <reference/Cache.h>
#include <reference/DepthFrame.h>
#include <reference/DepthStream.h>
#include <reference/Integrator.h>
#include <reference/Mesher.h>
#include <reference/Volume.h>

#include "util.h"



BOOST_AUTO_TEST_SUITE( Mesher )

BOOST_AUTO_TEST_CASE( Triangulate )
{
	/*
	Quick visual test to verify triangulation works correctly:
	One depth frame is integrated (generated with poly2depth) and then
	the volume is triangulated using mc and stored as an .obj
	*/
	svc::Integrator i;
	svc::Mesher m;

	svc::Volume v( 512, 2.0f, 2, 0.02f );
	svc::Cache cache;

	svc::DepthStream ds( ( boost::filesystem::current_path() / "content/imrod_v2.depth" ).string().c_str() );

	svc::DepthFrame depth;
	flink::float4x4 view, viewProj, viewToWorld;
	flink::float4 eye, forward;

	ds.NextFrame( depth, view );
	ComputeMatrices( view, eye, forward, viewProj, viewToWorld );

	i.Integrate( v, cache, depth, eye, forward, viewProj, viewToWorld );

	flink::vector< flink::float4 > VB;
	flink::vector< unsigned > IB;
	m.Triangulate( v, cache, VB, IB );
	m.Triangulate( v, cache, VB, IB );
	svc::Mesher::Mesh2Obj( VB, IB, "C:/TEMP/volume_triangulate.obj" );

	BOOST_REQUIRE( true );
}

BOOST_AUTO_TEST_SUITE_END()