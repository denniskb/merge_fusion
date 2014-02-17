#include <boost/test/auto_unit_test.hpp>

#include <cstdio>
#include <vector>

#include <DirectXMath.h>

#include <boost/filesystem/operations.hpp>

#include <server/DepthFrame.h>
#include <server/DepthStream.h>
#include <server/flink.h>
#include <server/Volume.h>
#include <server/Voxel.h>

using namespace flink;



#define RUN_TESTS 0



#if RUN_TESTS

static void ComputeMatrices
(
	float4x4 const & view,
	float4 & outEye,
	float4 & outForward,
	float4x4 & outViewProj
)
{
	matrix _view = load( & view );
	matrix _viewInv = XMMatrixInverse( nullptr, _view );
	vector _eye = set( 0.0f, 0.0f, 0.0f, 1.0f ) * _viewInv;
	vector _forward = set( 0.0f, 0.0f, -1.0f, 0.0f ) * _viewInv;

	matrix _viewProj = _view * XMMatrixPerspectiveFovRH( 0.778633444f, 4.0f / 3.0f, 0.8f, 4.0f );

	outEye = store( _eye );
	outForward = store( _forward );
	outViewProj = store( _viewProj );
}

#endif



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

#if RUN_TESTS

	kppl::Volume v( 256, 2.0f, 0.04f );
	kppl::DepthStream ds( ( boost::filesystem::current_path() / "content/imrod_v2.depth" ).string().c_str() );

	kppl::DepthFrame depth;
	float4x4 view, viewProj;
	float4 eye, forward;

	ds.NextFrame( depth, view );
	ComputeMatrices( view, eye, forward, viewProj );

	v.Integrate( depth, eye, forward, viewProj );

	FILE * debug;
	fopen_s( & debug, "C:/TEMP/volume_integrate.obj", "w" );

	for( int z = 0; z < v.Resolution(); z++ )
		for( int y = 0; y < v.Resolution(); y++ )
			for( int x = 0; x < v.Resolution(); x++ )
			{
				kppl::Voxel vx = v( x, y, z );
				if( vx.Weight() > 0 && abs( vx.Distance( 0.02f ) ) < 0.005f )
				{
					XMFLOAT4A pos = v.VoxelCenter( x, y, z );
					fprintf_s( debug, "v %f %f %f\n", pos.x, pos.y, pos.z );
				}
			}
	
	fclose( debug );

	BOOST_REQUIRE( true );

#endif
}

BOOST_AUTO_TEST_CASE( Triangulate )
{
	/*
	Quick visual test to verify triangulation works correctly:
	One depth frame is integrated (generated with poly2depth) and then
	the volume is triangulated using mc and stored as an .obj
	*/

#if RUN_TESTS

	kppl::Volume v( 256, 2.0f, 0.04f );
	kppl::DepthStream ds( ( boost::filesystem::current_path() / "content/imrod_v2.depth" ).string().c_str() );

	kppl::DepthFrame depth;
	float4x4 view, viewProj;
	float4 eye, forward;

	ds.NextFrame( depth, view );
	ComputeMatrices( view, eye, forward, viewProj );

	v.Integrate( depth, eye, forward, viewProj );
	v.Triangulate( "C:/TEMP/volume_triangulate.obj" );

	BOOST_REQUIRE( true );

#endif
}

BOOST_AUTO_TEST_SUITE_END()