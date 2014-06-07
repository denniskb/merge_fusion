#include <boost/filesystem/operations.hpp>
#include <boost/test/auto_unit_test.hpp>

#include <kifi/util/DirectXMathExt.h>
#include <kifi/util/vector2d.h>

#include <kifi/DepthStream.h>
#include <kifi/Integrator.h>
#include <kifi/Mesher.h>
#include <kifi/Volume.h>

#include <helper_test.h>

using namespace kifi;



BOOST_AUTO_TEST_SUITE( MesherTest )

BOOST_AUTO_TEST_CASE( Triangulate )
{
	/*
	Quick visual test to verify triangulation works correctly:
	One depth frame is integrated (generated with poly2depth) and then
	the volume is triangulated using mc and stored as an .obj
	*/
	Integrator i;
	Mesher m;

	Volume v( 256, 2.0f, 0.02f );

	DepthStream ds( ( boost::filesystem::current_path() / "../content/imrod_v2.depth" ).string().c_str() );

	util::vector2d< float > depth;
	util::float4x4 view, viewProj, viewToWorld;
	util::float4 eye, forward;

	ds.NextFrame( depth, view );
	ComputeMatrices( view, eye, forward, viewProj, viewToWorld );

	i.Integrate( v, depth, 2, eye, forward, viewProj, viewToWorld );

	std::vector< util::float4 > VB;
	std::vector< unsigned > IB;
	m.Triangulate( v, VB, IB );
	Mesher::Mesh2Obj( VB, IB, "C:/TEMP/volume_triangulate.obj" );

	BOOST_REQUIRE( true );
}

BOOST_AUTO_TEST_SUITE_END()