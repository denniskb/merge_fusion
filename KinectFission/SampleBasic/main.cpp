#include <vector>

#include <kifi/util/math.h>
#include <kifi/util/stop_watch.h>
#include <kifi/util/vector2d.h>

#include <kifi/DepthStream.h>
#include <kifi/Integrator.h>
#include <kifi/Mesher.h>
#include <kifi/Volume.h>

#include <UnitTests/helper_test.h>

using namespace kifi;
using namespace kifi::util;



int main()
{
	DepthStream depthStreamHouse( "I:/tmp/house.depth" );
	vector2d< float > synthDepthFrame;
	
	float4   eye;
	float4   forward;
	float4x4 view, viewToWorld, viewProj;

	Volume volume( 512, 4.0f, 0.02f );

	Integrator integrator;
	Mesher     mesher;

	std::vector< float3   > vertices;
	std::vector< unsigned > indices;

	depthStreamHouse.NextFrame( synthDepthFrame, view );
	ComputeMatrices( view, eye, forward, viewProj, viewToWorld );
	integrator.Integrate( volume, synthDepthFrame, eye, forward, viewProj, viewToWorld );
	
	for( int i = 0; i < 100; i++ ) 
	{
		depthStreamHouse.NextFrame( synthDepthFrame, view );
		ComputeMatrices( view, eye, forward, viewProj, viewToWorld );
		integrator.Integrate( volume, synthDepthFrame, eye, forward, viewProj, viewToWorld );
		//mesher.Mesh( volume, vertices, indices );
	}

	//mesher.Mesh( volume, vertices );
	mesher.Mesh( volume, vertices, indices );
	Mesher::Mesh2Obj( vertices, indices, "I:/tmp/house.obj" );

	return 0;
}