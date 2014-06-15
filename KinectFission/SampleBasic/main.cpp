#include <cstdio>
#include <vector>

#include <kifi/util/math.h>
#include <kifi/util/stop_watch.h>
#include <kifi/util/vector2d.h>

#include <kifi/DepthStream.h>
#include <kifi/Integrator.h>
#include <kifi/Mesher.h>
#include <kifi/Splatter.h>
#include <kifi/Volume.h>

#include <UnitTests/helper_test.h>

using namespace kifi;
using namespace kifi::util;



static void mesh2obj( std::vector< float4 > const & vertices, char const * outObjFileName )
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



int main()
{
	DepthStream depthStreamHouse( "C:/TEMP/house.depth" );
	vector2d< float > synthDepthFrame;
	
	float4   eye;
	float4   forward;
	float4x4 view;
	float4x4 viewProj;
	float4x4 viewToWorld;

	Volume volume( 512, 4.0f, 0.02f );

	Integrator integrator;



	std::printf( "%d total frames\n", depthStreamHouse.FrameCount() );

	for( int i = 0; i < 100; i++ )
	{
		depthStreamHouse.NextFrame( synthDepthFrame, view );
		ComputeMatrices( view, eye, forward, viewProj, viewToWorld );

		integrator.Integrate( volume, synthDepthFrame, eye, forward, viewProj, viewToWorld );
		std::printf( "Frame %d\n", i );
	}

	std::vector< float4 > vertices;
	//std::vector< unsigned > indices;
	
	//*/
	Splatter::Splat( volume, vertices );
	/*/
	Mesher mesher;
	mesher.Triangulate( volume, vertices, indices );
	//*/
	
	//Mesher::Mesh2Obj( vertices, indices, "C:/TEMP/house.obj" );

	// simple point cloud for debugging visually
	//for( auto it = volume.Data().keys_cbegin(), end = volume.Data().keys_cend(); it != end; ++it )
	//{
	//	unsigned x, y, z;
	//	unpack( * it, x, y, z );
	//	
	//	vertices.push_back( float4( (float)x, (float)y, (float)z, 1.0f ) );
	//}
	
	mesh2obj( vertices, "C:/TEMP/house.obj" );
	
	return 0;
}