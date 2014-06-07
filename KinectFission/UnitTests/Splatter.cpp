#include <vector>

#include <boost/filesystem/operations.hpp>
#include <boost/test/auto_unit_test.hpp>

#include <kifi/util/DirectXMathExt.h>
#include <kifi/util/vector2d.h>

#include <kifi/DepthStream.h>
#include <kifi/Integrator.h>
#include <kifi/Mesher.h>
#include <kifi/Volume.h>
#include <kifi/Splatter.h>

#include <helper_test.h>

using namespace kifi;



BOOST_AUTO_TEST_SUITE( Splatter )

BOOST_AUTO_TEST_CASE( Splat )
{
	/*
	Quick visual test to verify integration works correctly:
	One depth frame is integrated (generated with poly2depth) and then
	the volume is splatted and stored vertices as an .obj
	*/
	Volume v( 256, 2.0f, 0.02f );
	Integrator i;

	DepthStream ds( ( boost::filesystem::current_path() / "../content/imrod_v2.depth" ).string().c_str() );

	util::vector2d< float > depth;
	util::float4x4 view, viewProj, viewToWorld;
	util::float4 eye, forward;

	ds.NextFrame( depth, view );
	ComputeMatrices( view, eye, forward, viewProj, viewToWorld );

	i.Integrate( v, depth, 2, eye, forward, viewProj, viewToWorld );

	std::vector< util::float4 > verts;
	kifi::Splatter::Splat( v, verts );

	Mesher::Mesh2Obj( verts, std::vector< unsigned >(), "C:/TEMP/volume_splat.obj" );

	BOOST_REQUIRE( true );
}

BOOST_AUTO_TEST_SUITE_END()