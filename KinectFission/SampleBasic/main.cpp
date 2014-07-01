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



int main()
{
	DepthStream depthStreamHouse( "I:/tmp/house.depth" );
	vector2d< float > synthDepthFrame;
	
	vec3      eye;
	vec3      forward;
	matrix4x3 view, viewToWorld;
	matrix    viewProj;

	Volume volume( 512, 4.0f, 0.02f );

	Integrator integrator;

	std::vector< vec3 > vertices;



	depthStreamHouse.NextFrame( synthDepthFrame, view );
	ComputeMatrices( view, eye, forward, viewProj, viewToWorld );
	integrator.Integrate( volume, synthDepthFrame, eye, forward, viewProj, viewToWorld );
	
	for( int i = 0; i < 200; i++ ) 
	{
		depthStreamHouse.NextFrame( synthDepthFrame, view );
		ComputeMatrices( view, eye, forward, viewProj, viewToWorld );
		integrator.Integrate( volume, synthDepthFrame, eye, forward, viewProj, viewToWorld );
		//Splatter::Splat( volume, vertices );
	}

	Splatter::Splat( volume, vertices );

	mesh2obj( vertices, "I:/tmp/house.obj" );
	
	return 0;
}