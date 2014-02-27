#include <server/DepthStream.h>

#pragma warning( push )
#pragma warning( disable: 4100 4267 4515 4996 )

#include <server/DeviceDepthFrame.h>
#include <server/DeviceVolume.h>

#pragma warning( pop )

#include <server/flink.h>
#include <server/HostDepthFrame.h>

#include "util.h"



int main()
{
	kppl::DeviceVolume v( 256, 2.0f, 0.02f );
	kppl::DepthStream ds( "C:/TEMP/test.depth" );

	kppl::HostDepthFrame depth;
	kppl::DeviceDepthFrame ddepth( depth );

	flink::float4x4 view, viewProj;
	flink::float4 eye, forward;	
	int i = 0;
	while( ds.NextFrame( depth, view ) )
	{
		ComputeMatrices( view, eye, forward, viewProj );

		char fileName[ 64 ];
		sprintf_s( fileName, 64, "C:/TEMP/dennis%03d.obj", i );

		ddepth << depth;
		v.Integrate( ddepth, eye, forward, viewProj );
		v.Triangulate( fileName );
		
		i++;
	}
}