#include <vector>

#include <GL/glew.h>
#include <GL/GL.h>
#include <GL/freeglut.h>

#include <kifi/util/math.h>
#include <kifi/util/vector2d.h>

#include <kifi/DepthStream.h>
#include <kifi/Pipeline.h>
#include <kifi/Renderer.h>

#include <kifi/util/stop_watch.h>

using namespace kifi;
using namespace kifi::util;

#define BUFFER_OFFSET(i) ((char *)NULL + (i))



util::vector2d< int > backBuffer( 1024, 768 );

DepthStream depthStreamHouse( "I:/tmp/imrod.depth" );
//DepthStream depthStreamHouse( "I:/tmp/icp_test.depth" );
vector2d< float > synthDepthFrame;
//DepthSensorParams cameraParams( int2( 640, 480 ), float2( 585.0f ), float2( 320, 240 ), float2( 0.8f, 4.0f ) );
DepthSensorParams cameraParams( DepthSensorParams::KinectParams( KinectDepthSensorResolution640x480, KinectDepthSensorModeFar ) );

Pipeline pipeline( cameraParams, 512, 2.0f, 0.01f );

std::vector< VertexPositionNormal > vertices;
std::vector< unsigned > indices;

Renderer r;

unsigned vbID, ibID; // opengl vertex plus index buffer

//void myIdleFunc()
void myIdleFunc( int button, int state, int x, int y )
{
	float4x4 worldToEye;

	if( GLUT_MIDDLE_BUTTON == state )
		if( depthStreamHouse.NextFrame( synthDepthFrame, worldToEye ) )
		{
			pipeline.Integrate( synthDepthFrame, worldToEye );
			chrono::stop_watch sw;
			pipeline.Mesh( vertices, indices );
			//pipeline.Mesh( vertices );
			sw.take_time( "tmesh" );
			//sw.print_times();

			//r.Render( vertices, cameraParams.EyeToClipRH() * worldToEye, backBuffer );

			glutPostRedisplay();
		}
		else
			Mesher::Mesh2Obj( vertices, indices, "I:/tmp/imrod.obj" );
}

void myDisplayFunc()
{
	glClear( GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT );
	glEnable( GL_DEPTH_TEST );
	//glShadeModel( GL_FLAT );
	glShadeModel( GL_SMOOTH );
	
	/*glTexImage2D( GL_TEXTURE_2D, 0, 3, backBuffer.width(), backBuffer.height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, backBuffer.data() );

	glBegin( GL_QUADS );
        glTexCoord2d( 0.0, 0.0 ); glVertex2d( -1.0,  1.0 );
        glTexCoord2d( 1.0, 0.0 ); glVertex2d(  1.0,  1.0 );
        glTexCoord2d( 1.0, 1.0 ); glVertex2d(  1.0, -1.0 );
        glTexCoord2d( 0.0, 1.0 ); glVertex2d( -1.0, -1.0 );
    glEnd();*/

	auto tmp = invert_transform( pipeline.EyeToWorld() );
	auto m = cameraParams.EyeToClipRH() * tmp;

	glPushMatrix();
	glLoadMatrixf( reinterpret_cast< float * >( & m ) );
	
	glEnableClientState( GL_VERTEX_ARRAY );
	glVertexPointer( 3, GL_FLOAT, 24, vertices.data() );

	glEnableClientState( GL_COLOR_ARRAY );
	glColorPointer( 3, GL_FLOAT, 24, reinterpret_cast< float * >( vertices.data() ) + 3 );

	//glDrawElements( GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, indices.data() );
	glDrawArrays( GL_POINTS, 0, vertices.size() );

	glDisableClientState( GL_COLOR_ARRAY );
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

	glewInit();

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