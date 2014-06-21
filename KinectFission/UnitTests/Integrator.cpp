#include <cstdio>

#include <boost/filesystem/operations.hpp>
#include <boost/test/auto_unit_test.hpp>

#include <kifi/util/math.h>
#include <kifi/util/vector2d.h>

#include <kifi/DepthStream.h>
#include <kifi/Integrator.h>
#include <kifi/Volume.h>

#include <helper_test.h>

using namespace kifi;



BOOST_AUTO_TEST_SUITE( IntegratorTest )

BOOST_AUTO_TEST_CASE( Integrate )
{
	/*
	Quick visual test to verify integration works correctly:
	One depth frame is integrated (generated with poly2depth) and then
	all voxels near surface are stored as vertices to an .obj
	*/
	Integrator i;

	Volume v( 256, 2.0f, 0.02f );

	DepthStream ds( ( boost::filesystem::current_path() / "../content/imrod_v2.depth" ).string().c_str() );

	util::vector2d< float > depth;
	util::matrix4x3 view, viewToWorld;
	util::matrix viewProj;
	util::vec3 eye, forward;

	ds.NextFrame( depth, view );
	ComputeMatrices( view, eye, forward, viewProj, viewToWorld );

	i.Integrate( v, depth, eye, forward, viewProj, viewToWorld );

	FILE * debug;
	fopen_s( & debug, TMP_DIR "/volume_integrate.obj", "w" );

	for( auto it = v.Data().keys_cbegin(), end = v.Data().keys_cend(); it != end; ++it )
	{
		std::uint32_t x, y, z;
		util::unpack( * it, x, y, z );
		
		util::vec3 pos = v.VoxelCenter( x, y, z );
		
		fprintf_s( debug, "v %f %f %f\n", pos.x, pos.y, pos.z );
	}
	
	fclose( debug );

	BOOST_REQUIRE( true );
}

BOOST_AUTO_TEST_SUITE_END()