#include <vector>

#include <boost/filesystem/operations.hpp>
#include <boost/test/auto_unit_test.hpp>

#include <kifi/util/math.h>
#include <kifi/util/vector2d.h>

#include <kifi/DepthSensorParams.h>
#include <kifi/DepthStream.h>
#include <kifi/Integrator.h>
#include <kifi/Mesher.h>
#include <kifi/Volume.h>

using namespace kifi;



BOOST_AUTO_TEST_SUITE( MesherTest )

BOOST_AUTO_TEST_CASE( Mesh )
{
	/*
	Quick visual test to verify integration works correctly:
	One depth frame is integrated (generated with poly2depth) and then
	the volume is splatted and stored vertices as an .obj
	*/
	Volume v( 256, 2.0f, 0.02f );
	Integrator i;
	Mesher m;

	DepthStream ds( ( boost::filesystem::current_path() / "../content/imrod_v2.depth" ).string().c_str() );

	util::vector2d< float > depth;
	util::float4x4 view;

	ds.NextFrame( depth, view );
	i.Integrate( v, depth, DepthSensorParams::KinectParams( KinectDepthSensorResolution640x480, KinectDepthSensorModeFar ), view );

	std::vector< util::float3 > verts;
	std::vector< unsigned > indices;
	
	m.Mesh( v, verts );
	Mesher::Mesh2Obj( verts, indices, TMP_DIR "/volume_splat.obj" );

	m.Mesh( v, verts, indices );
	Mesher::Mesh2Obj( verts, indices, TMP_DIR "/volume_mesh.obj" );

	BOOST_REQUIRE( true );
}

BOOST_AUTO_TEST_SUITE_END()