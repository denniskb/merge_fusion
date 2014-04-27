#include <boost/test/auto_unit_test.hpp>

#include <vector>

#include <boost/filesystem/operations.hpp>

#include <reference/DepthStream.h>
#include <reference/dxmath.h>
#include <reference/Integrator.h>
#include <reference/Mesher.h>
#include <reference/vector2d.h>
#include <reference/Volume.h>
#include <reference/Splatter.h>

#include "util.h"



BOOST_AUTO_TEST_SUITE( Splatter )

BOOST_AUTO_TEST_CASE( Splat )
{
	/*
	Quick visual test to verify integration works correctly:
	One depth frame is integrated (generated with poly2depth) and then
	the volume is splatted and stored vertices as an .obj
	*/
	svc::Volume v( 256, 2.0f, 0.02f );
	svc::Integrator i;

	svc::DepthStream ds( ( boost::filesystem::current_path() / "../content/imrod_v2.depth" ).string().c_str() );

	svc::vector2d< float > depth;
	svc::float4x4 view, viewProj, viewToWorld;
	svc::float4 eye, forward;

	ds.NextFrame( depth, view );
	ComputeMatrices( view, eye, forward, viewProj, viewToWorld );

	i.Integrate( v, depth, 2, eye, forward, viewProj, viewToWorld );

	std::vector< svc::float4 > verts;
	svc::Splatter::Splat( v, verts );
	//svc::Splatter::Splat( v, verts );

	svc::Mesher::Mesh2Obj( verts, std::vector< unsigned >(), "C:/TEMP/volume_splat.obj" );

	BOOST_REQUIRE( true );
}

BOOST_AUTO_TEST_SUITE_END()