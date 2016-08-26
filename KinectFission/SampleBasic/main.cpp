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

DepthStream depthStreamHouse( "D:/Desktop/house.depth" );
vector2d< float > synthDepthFrame;
DepthSensorParams cameraParams( int2( 640, 480 ), float2( 571.26f ), float2( 320, 240 ), float2( 0.4f, 4.0f ) );
//DepthSensorParams cameraParams( DepthSensorParams::KinectParams( KinectDepthSensorResolution640x480, KinectDepthSensorModeFar ) );

Pipeline pipeline( cameraParams, 512, 2.0f, 0.02f );

std::vector< float3   > vertices;
std::vector< unsigned > indices;

Renderer r;

//void myIdleFunc()
void myIdleFunc( int button, int state, int x, int y )
{
	float4x4 worldToEye;

	if( GLUT_MIDDLE_BUTTON == state )
		if( depthStreamHouse.NextFrame( synthDepthFrame, worldToEye ) )
		{
			pipeline.Integrate( synthDepthFrame, worldToEye );
			pipeline.Mesh( vertices, indices );
			//pipeline.Mesh( vertices );

			//float4x4 tmp = pipeline.EyeToWorld(); invert_transform( tmp );
			//r.Render( vertices, cameraParams.EyeToClipRH() * tmp, backBuffer );
			//r.Render( vertices, cameraParams.EyeToClipRH() * worldToEye, backBuffer );

			glutPostRedisplay();
		}
		else
			Mesher::Mesh2Obj( vertices, indices, "D:/Desktop/house.obj" );
}

void myDisplayFunc()
{
	glClear( GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT );
	glEnable( GL_DEPTH_TEST );

	/*glTexImage2D( GL_TEXTURE_2D, 0, 3, backBuffer.width(), backBuffer.height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, backBuffer.data() );

	glBegin( GL_QUADS );
        glTexCoord2d( 0.0, 0.0 ); glVertex2d( -1.0,  1.0 );
        glTexCoord2d( 1.0, 0.0 ); glVertex2d(  1.0,  1.0 );
        glTexCoord2d( 1.0, 1.0 ); glVertex2d(  1.0, -1.0 );
        glTexCoord2d( 0.0, 1.0 ); glVertex2d( -1.0, -1.0 );
    glEnd();*/

	auto tmp = pipeline.EyeToWorld(); invert_transform( tmp );
	auto m = cameraParams.EyeToClipRH() * tmp;

	glPushMatrix();
	glLoadMatrixf( reinterpret_cast< float * >( & m ) );
	
	glEnableClientState( GL_VERTEX_ARRAY );
	glVertexPointer( 3, GL_FLOAT, sizeof( float3 ), vertices.data() );

	glDrawArrays( GL_POINTS, 0, vertices.size() );

	glDisableClientState( GL_VERTEX_ARRAY );

	glPopMatrix();

    glutSwapBuffers();
}

int main( int argc, char ** argv )
{
	glutInit( & argc, argv );

	glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH );
	glutInitWindowPosition( 250, 80 );
	glutInitWindowSize( backBuffer.width(), backBuffer.height() );
	glutCreateWindow( "KinectFission" );

	glutDisplayFunc( myDisplayFunc );
	glutMouseFunc( myIdleFunc );
	//glutIdleFunc( myIdleFunc );

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