#include <fstream>

#include <boost/filesystem/operations.hpp>
#include <boost/test/auto_unit_test.hpp>

#include <kifi/util/math.h>
#include <kifi/util/vector2d.h>

#include <kifi/DepthStream.h>
#include <kifi/Integrator.h>
#include <kifi/Volume.h>

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
	util::float4x4 view;

	ds.NextFrame( depth, view );
	i.Integrate( v, depth, DepthSensorParams::KinectV1Params( KinectDepthSensorResolution640x480, KinectDepthSensorModeFar ), view );

	std::ofstream debug( TMP_DIR "/volume_integrate.obj" );
	if( ! debug )
		return;

	for( auto it = v.Data().keys_cbegin(), end = v.Data().keys_cend(); it != end; ++it )
	{
		unsigned x, y, z;
		util::unpack( * it, x, y, z );
		
		util::float3 pos = v.VoxelCenter( x, y, z );
		
		debug << "v " << pos.x() << " " << pos.y() << " " << pos.z() << "\n";
	}
	
	BOOST_REQUIRE( true );
}

BOOST_AUTO_TEST_SUITE_END()