#include <vector>

#include <boost/filesystem/operations.hpp>
#include <boost/test/auto_unit_test.hpp>

#include <kifi/util/math.h>
#include <kifi/util/vector2d.h>

#include <kifi/DepthStream.h>
#include <kifi/Integrator.h>
#include <kifi/Mesher.h>
#include <kifi/Volume.h>
#include <kifi/Splatter.h>

#include <helper_test.h>

using namespace kifi;



static void mesh2obj( std::vector< util::vec3 > const & vertices, char const * outObjFileName )
{
	FILE * file;
	fopen_s( & file, outObjFileName, "w" );

	for( int i = 0; i < vertices.size(); i++ )
	{
		auto v = vertices[ i ];
		fprintf_s( file, "v %f %f %f\n", v.x, v.y, v.z );
	}
	
	fclose( file );
}



BOOST_AUTO_TEST_SUITE( SplatterTest )

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
	util::matrix4x3 view, viewToWorld;
	util::matrix viewProj;
	util::vec3 eye, forward;

	ds.NextFrame( depth, view );
	ComputeMatrices( view, eye, forward, viewProj, viewToWorld );

	i.Integrate( v, depth, eye, forward, viewProj, viewToWorld );

	std::vector< util::vec3 > verts;
	Splatter::Splat( v, verts );

	mesh2obj( verts, "C:/TEMP/volume_splat.obj" );

	BOOST_REQUIRE( true );
}

BOOST_AUTO_TEST_SUITE_END()