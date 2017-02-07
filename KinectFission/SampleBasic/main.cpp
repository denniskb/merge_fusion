#include <vector>

#include <GL/freeglut.h>
#include <GL/GL.h>

#include <kifi/util/math.h>
#include <kifi/util/vector2d.h>

#include <kifi/DepthStream.h>
#include <kifi/Pipeline.h>
#include <kifi/Renderer.h>

using namespace kifi;
using namespace kifi::util;

// HACK
#include <kifi/util/stop_watch.h>



util::vector2d< int > backBuffer( 640, 480 );
vector2d< float > synthDepthFrame;

#if 1
	DepthStream depthStreamHouse( "D:/Desktop/synth_noise.depth" );
	DepthSensorParams cameraParams = DepthSensorParams::KinectParams( KinectDepthSensorResolution640x480, KinectDepthSensorModeFar );
#else
	DepthStream depthStreamHouse( "D:/Desktop/raw_unknown_pose.depth" );
	DepthSensorParams cameraParams = DepthSensorParams::KinectV2Params();
#endif

Pipeline pipeline( cameraParams, 1024, 4.0f, 0.04f );

std::vector< float3   > vertices;
std::vector< unsigned > indices;

Renderer r;

void myIdleFunc()
{
	static bool done = false;

	float4x4 worldToEye;

	if( depthStreamHouse.NextFrame( synthDepthFrame, worldToEye ) )
	{			
		std::printf( "#points: %.2fM\n", pipeline.Volume().Data().size() / 1000000.0 );
		pipeline.Integrate( synthDepthFrame, worldToEye );
		
		util::chrono::stop_watch sw;
		r.Render( pipeline.Volume(), cameraParams.EyeToClipRH() * worldToEye, backBuffer );
		sw.take_time( "tRender" );
		sw.print_times();
			
		std::printf( "\n" );

		glutPostRedisplay();
	}
	else if( ! done )
	{
		pipeline.Mesh( vertices, indices );
		Mesher::Mesh2Obj( vertices, indices, "D:/Desktop/house.obj" );
		std::printf( "DONE!\n" );
		done = true;
	}
}

void myDisplayFunc()
{
	glClear( GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT );
	glEnable( GL_DEPTH_TEST );

	glTexImage2D( GL_TEXTURE_2D, 0, 3, (GLsizei) backBuffer.width(), (GLsizei) backBuffer.height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, backBuffer.data() );

	glBegin( GL_QUADS );
        glTexCoord2d( 0.0, 0.0 ); glVertex2d( -1.0,  1.0 );
        glTexCoord2d( 1.0, 0.0 ); glVertex2d(  1.0,  1.0 );
        glTexCoord2d( 1.0, 1.0 ); glVertex2d(  1.0, -1.0 );
        glTexCoord2d( 0.0, 1.0 ); glVertex2d( -1.0, -1.0 );
    glEnd();

	auto tmp = pipeline.EyeToWorld(); invert_transform( tmp );
	auto m = cameraParams.EyeToClipRH() * tmp;

    glutSwapBuffers();
}

int main( int argc, char ** argv )
{
	glutInit( & argc, argv );

	glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH );
	glutInitWindowPosition( 250, 80 );
	glutInitWindowSize( (int) backBuffer.width(), (int) backBuffer.height() );
	glutCreateWindow( "KinectFission" );

	glutDisplayFunc( myDisplayFunc );
	glutIdleFunc( myIdleFunc );

	// Set up the texture
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP); 
 
	// Enable textures
	glEnable(GL_TEXTURE_2D);

	glutMainLoop();

	return 0;
}