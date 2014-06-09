#include <cstdio>
#include <vector>

#include <kifi/util/DirectXMathExt.h>
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

	chrono::stop_watch sw;



	std::printf( "%d total frames\n", depthStreamHouse.FrameCount() );

	for( int i = 0; i < 200; i++ )
	{
		depthStreamHouse.NextFrame( synthDepthFrame, view );
		ComputeMatrices( view, eye, forward, viewProj, viewToWorld );

		sw.restart();
		integrator.Integrate( volume, synthDepthFrame, 2, eye, forward, viewProj, viewToWorld );
		float t = sw.elapsed_milliseconds();

		std::printf( "Frame %4d, nVoxels: %4dk, t: %.1fms\n", i, volume.Data().size() * 8 / 1000, t );
	}

	std::vector< float4 > vertices;
	std::vector< unsigned > indices;

	/*/
	Splatter::Splat( volume, vertices );
	/*/
	Mesher mesher;
	mesher.Triangulate( volume, vertices, indices );
	//*/

	Mesher::Mesh2Obj( vertices, indices, "C:/TEMP/house.obj" );

	return 0;
}