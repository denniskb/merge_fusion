#include <memory>
#include <vector>

#include <GL/glew.h>
#include <GL/GL.h>
#include <GL/freeglut.h>

#include <kifi/util/math.h>
#include <kifi/util/vector2d.h>

#include <kifi/DepthStream.h>
#include <kifi/Pipeline.h>

using namespace kifi;
using namespace kifi::util;



std::unique_ptr< DepthStream > depthStream;
vector2d< float > synthDepthFrame;
DepthSensorParams cameraParams( DepthSensorParams::KinectParams( KinectDepthSensorResolution640x480, KinectDepthSensorModeFar ) );

std::unique_ptr< Pipeline > pipeline;

std::vector< VertexPositionNormal > vertices;
std::vector< unsigned > indices;

bool triangles = true;



void myMouseFunc( int button, int state, int x, int y )
{
	float4x4 worldToEye;

	if( GLUT_MIDDLE_BUTTON == state )
		if( depthStream->NextFrame( synthDepthFrame, worldToEye ) )
		{
			static int iFrame = 0;
			std::printf( "- Frame %d -\n", iFrame++ );

			pipeline->Integrate( synthDepthFrame );
			
			if( triangles )
				pipeline->Mesh( vertices, indices );

			std::printf( "\n" );

			glutPostRedisplay();
		}
#if 1
		else
			glutExit();
#else
		else
			Mesher::Mesh2Obj( vertices, indices, "I:/tmp/imrod.obj" );
#endif
}

void myDisplayFunc()
{
	glClear( GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT );
	glEnable( GL_DEPTH_TEST );

	auto m = cameraParams.EyeToClipRH() * invert_transform( pipeline->EyeToWorld() );

	glPushMatrix();
	glLoadMatrixf( reinterpret_cast< float * >( & m ) );
	
	glEnableClientState( GL_VERTEX_ARRAY );
	glVertexPointer( 3, GL_FLOAT, 24, pipeline->SynthPointCloud().data() );

	glEnableClientState( GL_COLOR_ARRAY );
	glColorPointer( 3, GL_FLOAT, 24, reinterpret_cast< float const * >( pipeline->SynthPointCloud().data() ) + 3 );

	if( triangles )
		glDrawElements( GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, indices.data() );
	else
		glDrawArrays( GL_POINTS, 0, pipeline->SynthPointCloud().size() );

	glDisableClientState( GL_COLOR_ARRAY );
	glDisableClientState( GL_VERTEX_ARRAY );

	glPopMatrix();

    glutSwapBuffers();
}

#include <direct.h>

int main( int argc, char ** argv )
{
	if( 3 != argc )
	{
		std::printf
		(
			"Usage:   SampleBasic.exe input resolution\n"
			"Example: SampleBasic.exe imrod.depth 512\n"
		);
		
		return 1;
	}

	glutInit( & argc, argv );

	glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH );
	glutInitWindowPosition( 250, 80 );
	glutInitWindowSize( 1024, 768 );
	glutCreateWindow( "KinectFission" );

	glewInit();

	glutDisplayFunc( myDisplayFunc );
	glutMouseFunc( myMouseFunc );

	char workingDirectory[ 256 ];
	_getcwd( workingDirectory, sizeof( workingDirectory ) );

	char depthStreamPath[ 256 ];

	if( ':' == argv[ 1 ][ 1 ] ) // absolute path
		std::sprintf( depthStreamPath, "%s", argv[ 1 ] );
	else // relative path
		std::sprintf( depthStreamPath, "%s\\%s", workingDirectory, argv[ 1 ] );

	depthStream.reset( new DepthStream( depthStreamPath ) );
	pipeline.reset( new Pipeline( cameraParams, atoi( argv[ 2 ] ), 2.0f, 0.02f ) );

	glutMainLoop();

	return 0;
}