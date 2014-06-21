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



static void mesh2obj( std::vector< vec3 > const & vertices, char const * outObjFileName )
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
	DepthStream depthStreamHouse( "I:/tmp/house.depth" );
	vector2d< float > synthDepthFrame;
	
	vec3      eye;
	vec3      forward;
	matrix4x3 view, viewToWorld;
	matrix    viewProj;

	Volume volume( 512, 4.0f, 0.02f );

	Integrator integrator;



	depthStreamHouse.NextFrame( synthDepthFrame, view );
	ComputeMatrices( view, eye, forward, viewProj, viewToWorld );
	
	for( int i = 0; i < 100; i++ ) 
	{
		//depthStreamHouse.NextFrame( synthDepthFrame, view );
		//ComputeMatrices( view, eye, forward, viewProj, viewToWorld );

		integrator.Integrate( volume, synthDepthFrame, eye, forward, viewProj, viewToWorld );
	}

	std::vector< vec3 > vertices;
	
#if 1
	Splatter::Splat( volume, vertices );
#else
	for( auto it = volume.Data().keys_cbegin(), end = volume.Data().keys_cend(); it != end; ++it )
	{
		unsigned x, y, z;
		unpack( * it, x, y, z );
		
		vertices.push_back( vec3( (float) x, (float) y, (float) z ) );
	}
#endif

	mesh2obj( vertices, "I:/tmp/house.obj" );
	
	return 0;
}