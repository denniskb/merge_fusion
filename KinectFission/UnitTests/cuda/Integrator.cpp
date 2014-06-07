#include <cstdio>

#include <boost/filesystem/operations.hpp>
#include <boost/test/auto_unit_test.hpp>

#include <cuda/DepthFrame.h>
#include <cuda/Integrator.h>
#include <cuda/Volume.h>

#include <dlh/DirectXMathExt.h>
#include <dlh/vector2d.h>

#include <reference/DepthStream.h>

#include "util.h"



BOOST_AUTO_TEST_SUITE( Integrator )

BOOST_AUTO_TEST_CASE( Integrate )
{
	/*
	Quick visual test to verify integration works correctly:
	One depth frame is integrated (generated with poly2depth) and then
	all voxels near surface are stored as vertices to an .obj
	*/
	svcu::Integrator i;

	svcu::Volume v( 256, 2.0f, 0.02f );

	svc::DepthStream ds( ( boost::filesystem::current_path() / "../content/imrod_v2.depth" ).string().c_str() );

	dlh::vector2d< float > depth;
	dlh::float4x4 view, viewProj, viewToWorld;
	dlh::float4 eye, forward;

	ds.NextFrame( depth, view );
	ComputeMatrices( view, eye, forward, viewProj, viewToWorld );

	svcu::DepthFrame ddepth;
	ddepth.CopyFrom( depth );

	i.Integrate( v, ddepth, 2, eye, forward, viewProj, viewToWorld );

	// TODO: Finish: Copy data back from GPU and output a debug .obj

	BOOST_REQUIRE( true );
}

BOOST_AUTO_TEST_SUITE_END()