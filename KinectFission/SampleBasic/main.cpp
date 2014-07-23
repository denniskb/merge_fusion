#include <vector>

#include <kifi/util/math.h>
#include <kifi/util/vector2d.h>

#include <kifi/DepthStream.h>
#include <kifi/Mesher.h>
#include <kifi/Pipeline.h>

using namespace kifi;
using namespace kifi::util;



int main()
{
	DepthStream depthStreamHouse( "I:/tmp/house.depth" );
	vector2d< float > synthDepthFrame;	
	float4x4 worldToEye;

	Pipeline pipeline( DepthSensorParams::KinectParams( KinectDepthSensorResolution640x480, KinectDepthSensorModeFar ) );

	for( int i = 0; i < 100; i++ ) 
	{
		depthStreamHouse.NextFrame( synthDepthFrame, worldToEye );
		pipeline.Integrate( synthDepthFrame, worldToEye );
	}

	Mesher mesher;
	std::vector< float3   > vertices;
	std::vector< unsigned > indices;

	mesher.Mesh( pipeline.Volume(), vertices, indices );
	
	Mesher::Mesh2Obj( vertices, indices, "I:/tmp/house.obj" );

	return 0;
}