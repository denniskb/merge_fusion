#include <cstdio>

#include <boost/filesystem/operations.hpp>
#include <boost/test/auto_unit_test.hpp>

#include <kifi/cuda/DepthFrame.h>
#include <kifi/cuda/Integrator.h>
#include <kifi/cuda/Volume.h>

#include <kifi/util/DirectXMathExt.h>
#include <kifi/util/vector2d.h>

#include <kifi/DepthStream.h>

#include <helper_test.h>

using namespace kifi;



BOOST_AUTO_TEST_SUITE( cuda_test )
BOOST_AUTO_TEST_SUITE( IntegratorTest )

BOOST_AUTO_TEST_CASE( Integrate )
{
	/*
	Quick visual test to verify integration works correctly:
	One depth frame is integrated (generated with poly2depth) and then
	all voxels near surface are stored as vertices to an .obj
	*/
	cuda::Integrator i;

	cuda::Volume v( 256, 2.0f, 0.02f );

	DepthStream ds( ( boost::filesystem::current_path() / "../content/imrod_v2.depth" ).string().c_str() );

	util::vector2d< float > depth;
	util::float4x4 view, viewProj, viewToWorld;
	util::float4 eye, forward;

	ds.NextFrame( depth, view );
	ComputeMatrices( view, eye, forward, viewProj, viewToWorld );

	cuda::DepthFrame ddepth;
	ddepth.CopyFrom( depth );

	i.Integrate( v, ddepth, 2, eye, forward, viewProj, viewToWorld );

	// TODO: Finish: Copy data back from GPU and output a debug .obj

	BOOST_REQUIRE( true );
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()