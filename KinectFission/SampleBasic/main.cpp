#include <vector>

#include <GL/glew.h>
#include <GL/GL.h>
#include <GL/freeglut.h>

#include <kifi/util/math.h>
#include <kifi/util/vector2d.h>

#include <kifi/DepthStream.h>
#include <kifi/Pipeline.h>
#include <kifi/Renderer.h>

using namespace kifi;
using namespace kifi::util;



DepthStream depthStreamHouse( "I:/tmp/imrod.depth" );
vector2d< float > synthDepthFrame;
DepthSensorParams cameraParams( DepthSensorParams::KinectParams( KinectDepthSensorResolution640x480, KinectDepthSensorModeFar ) );

Pipeline pipeline( cameraParams, 512, 2.0f, 0.01f );

std::vector< VertexPositionNormal > vertices;
std::vector< unsigned > indices;

bool triangles = false;



void myIdleFunc( int button, int state, int x, int y )
{
	float4x4 worldToEye;

	if( GLUT_MIDDLE_BUTTON == state )
		if( depthStreamHouse.NextFrame( synthDepthFrame, worldToEye ) )
		{
			pipeline.Integrate( synthDepthFrame, worldToEye );
			
			if( triangles )
				pipeline.Mesh( vertices, indices );
			else
				pipeline.Mesh( vertices );

			glutPostRedisplay();
		}
		else
			Mesher::Mesh2Obj( vertices, indices, "I:/tmp/imrod.obj" );
}

void myDisplayFunc()
{
	glClear( GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT );
	glEnable( GL_DEPTH_TEST );

	auto m = cameraParams.EyeToClipRH() * invert_transform( pipeline.EyeToWorld() );

	glPushMatrix();
	glLoadMatrixf( reinterpret_cast< float * >( & m ) );
	
	glEnableClientState( GL_VERTEX_ARRAY );
	glVertexPointer( 3, GL_FLOAT, 24, vertices.data() );

	glEnableClientState( GL_COLOR_ARRAY );
	glColorPointer( 3, GL_FLOAT, 24, reinterpret_cast< float * >( vertices.data() ) + 3 );

	if( triangles )
		glDrawElements( GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, indices.data() );
	else
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
	glutInitWindowSize( 1024, 768 );
	glutCreateWindow( "KinectFission" );

	glewInit();

	glutDisplayFunc( myDisplayFunc );
	glutMouseFunc( myIdleFunc );

	glutMainLoop();

	return 0;
}