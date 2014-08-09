#include <vector>

#include <GL/freeglut.h>
#include <GL/GL.h>

#include <kifi/util/math.h>
#include <kifi/util/vector2d.h>

#include <kifi/DepthStream.h>
#include <kifi/Pipeline.h>
#include <kifi/Renderer.h>

#include <kifi/util/stop_watch.h>

using namespace kifi;
using namespace kifi::util;



util::vector2d< int > backBuffer( 1024, 768 );

DepthStream depthStreamHouse( "I:/tmp/imrod.depth" );
vector2d< float > synthDepthFrame;
//DepthSensorParams cameraParams( int2( 640, 480 ), float2( 585.0f ), float2( 320, 240 ), float2( 0.8f, 4.0f ) );
DepthSensorParams cameraParams( DepthSensorParams::KinectParams( KinectDepthSensorResolution640x480, KinectDepthSensorModeFar ) );

Pipeline pipeline( cameraParams, 512, 4.0f, 0.01f );

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
			chrono::stop_watch sw;
			//pipeline.Mesh( vertices, indices );
			pipeline.Mesh( vertices );
			sw.take_time( "tmesh" );
			//sw.print_times();

			float4x4 tmp = pipeline.EyeToWorld(); invert_transform( tmp );
			//r.Render( vertices, cameraParams.EyeToClipRH() * tmp, backBuffer );
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
	glDisable( GL_CULL_FACE );
	
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
	
	/*glBegin( GL_TRIANGLES );
		for( int i = 0; i < indices.size(); i++ )
		{
			auto v = vertices[ indices[ i ] ];
			glVertex3f( v.x, v.y, v.z );
			
			auto vproj = m * float4( v, 1.0f );
			//float gray = 1.0f - (vproj.w - 0.5f) * 0.25f;
			float gray = (vproj.w - 0.9f) * 1.0f;
			glColor3f( gray, gray, gray );
		}
	glEnd();*/

	glPointSize( 1.0f );
	glBegin( GL_POINTS );
		for( int i = 0; i < vertices.size(); i++ )
		{
			auto v = vertices[ i ];
			glVertex3f( v.x, v.y, v.z );
			
			float gray = 1.0f;
			glColor3f( gray, gray, gray );
		}
	glEnd();

	glPopMatrix();

    glutSwapBuffers();
}

int main( int argc, char ** argv )
{
#if 1
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
#else	
	DepthStream depthStreamHouse( "I:/tmp/house.depth" );
	vector2d< float > synthDepthFrame;	
	float4x4 worldToEye;

	Pipeline pipeline( DepthSensorParams( int2( 640, 480 ), float2( 585.0f ), float2( 320, 240 ), float2( 0.8f, 4.0f ) ) );

	for( int i = 0; i < 200; i++ ) 
	{
		depthStreamHouse.NextFrame( synthDepthFrame, worldToEye );
		pipeline.Integrate( synthDepthFrame, worldToEye );
	}

	std::vector< float3   > vertices;
	std::vector< unsigned > indices;

	pipeline.Mesh( vertices, indices );
	
	Mesher::Mesh2Obj( vertices, indices, "I:/tmp/house.obj" );

	return 0;
#endif
}