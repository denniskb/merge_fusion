#include <vector>

#include <kifi/util/math.h>
#include <kifi/util/stop_watch.h>
#include <kifi/util/vector2d.h>

#include <kifi/DepthStream.h>
#include <kifi/Integrator.h>
#include <kifi/Mesher.h>
#include <kifi/Volume.h>

using namespace kifi;
using namespace kifi::util;



int main()
{
	DepthStream depthStreamHouse( "I:/tmp/house.depth" );
	vector2d< float > synthDepthFrame;
	
	float4x4 worldToEye;
	DepthSensorParams cameraParams = DepthSensorParams::KinectParams( KinectDepthSensorResolution640x480, KinectDepthSensorModeFar );

	Volume volume( 512, 4.0f, 0.02f );

	Integrator integrator;
	Mesher     mesher;

	std::vector< float3   > vertices;
	std::vector< unsigned > indices;

	depthStreamHouse.NextFrame( synthDepthFrame, worldToEye );

	for( int i = 0; i < 100; i++ ) 
	{
		depthStreamHouse.NextFrame( synthDepthFrame, worldToEye );
		integrator.Integrate( volume, synthDepthFrame, cameraParams, worldToEye );
		//mesher.Mesh( volume, vertices, indices );
	}

	//mesher.Mesh( volume, vertices );
	mesher.Mesh( volume, vertices, indices );
	Mesher::Mesh2Obj( vertices, indices, "I:/tmp/house.obj" );

	return 0;
}