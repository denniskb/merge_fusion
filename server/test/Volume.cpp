#include <boost/test/auto_unit_test.hpp>

#include <cstdio>
#include <ctime>
#include <vector>

#include <DirectXMath.h>

#include <server/DepthStream.h>
#include <server/Volume.h>
#include <server/Voxel.h>

using namespace DirectX;



BOOST_AUTO_TEST_SUITE( Volume )

BOOST_AUTO_TEST_CASE( getset )
{
	kppl::Volume vol( 10, 1.0f, 0.02f );
	BOOST_REQUIRE( 10 == vol.Resolution() );

	kppl::Voxel v;
	BOOST_REQUIRE( vol( 0, 0, 0 ) == v );
	BOOST_REQUIRE( vol( 3, 1, 9 ) == v );

	v.Update( 1.0f, 1.0f );
	BOOST_REQUIRE( vol( 2, 7, 1 ) != v );

	vol( 1, 1, 1 ) = v;
	BOOST_REQUIRE( vol( 1, 1, 1 ) == v );

	v.Update( 0.0f, 1.0f );
	vol( 1, 1, 1 ).Update( 0.0f, 1.0f );
	BOOST_REQUIRE( vol( 1, 1, 1 ) == v );
}

BOOST_AUTO_TEST_CASE( Integrate )
{
	/*
	Quick visual test to verify integration works correctly:
	One depth frame is integrated (generated with poly2depth) and then
	all voxels near surface are stored as vertices to an .obj
	*/

#if 0

	kppl::Volume v( 256, 2.0f, 0.02f );
	kppl::DepthStream ds( "C:/TEMP/debug.depth" );

	XMMATRIX _proj = XMMatrixPerspectiveFovRH( 0.778633444f, 4.0f / 3.0f, 0.8f, 4.0f );
	XMFLOAT4X4A proj;
	XMStoreFloat4x4A( & proj, _proj );

	std::vector< short > depth;
	XMFLOAT4X4A view;
	ds.NextFrame( depth, view );
	v.Integrate( depth, view, proj );

	FILE * debug;
	fopen_s( & debug, "C:/TEMP/integration_debug.obj", "w" );

	for( int z = 0; z < 256; z++ )
		for( int y = 0; y < 256; y++ )
			for( int x = 0; x < 256; x++ )
			{
				kppl::Voxel vx = v( x, y, z );
				if( vx.Weight() > 0 && abs( vx.Distance( 0.02f ) ) < 0.005f )
				{
					XMFLOAT4A pos = v.VoxelCenter( x, y, z );
					fprintf_s( debug, "v %f %f %f\n", pos.x, pos.y, pos.z );
				}
			}
	
	fclose( debug );

#endif

	BOOST_REQUIRE( true );
}

BOOST_AUTO_TEST_CASE( Triangulate )
{
	/*
	Quick visual test to verify triangulation works correctly:
	One depth frame is integrated (generated with poly2depth) and then
	the volume is triangulated using mc and stored as an .obj
	*/

#if 0

	kppl::Volume v( 256, 2.0f, 0.04f );
	kppl::DepthStream ds( "C:/TEMP/debug.depth" );

	XMMATRIX _proj = XMMatrixPerspectiveFovRH( 0.778633444f, 4.0f / 3.0f, 0.8f, 4.0f );
	XMFLOAT4X4A proj;
	XMStoreFloat4x4A( & proj, _proj );

	std::vector< short > depth;
	XMFLOAT4X4A view;
	ds.NextFrame( depth, view );
	v.Integrate( depth, view, proj );
	v.Triangulate( "C:/TEMP/triangulation_debug.obj" );

#endif
}

BOOST_AUTO_TEST_SUITE_END()