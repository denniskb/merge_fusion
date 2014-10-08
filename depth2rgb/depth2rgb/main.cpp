#include <cstdio>
#include <functional>
#include <map>
#include <vector>

static const unsigned KINECT_RES = 640 * 480;

std::vector< unsigned short > read_pgm( char const * file )
{
	std::vector< unsigned short > result( KINECT_RES );

	FILE * src = fopen( file, "rb" );
	fseek( src, 17, SEEK_SET );
	fread( result.data(), 2, KINECT_RES, src );
	fclose( src );

	return result;
}

std::vector< unsigned char > read_ppm( char const * file )
{
	std::vector< unsigned char > result( KINECT_RES * 3 );

	FILE * src = fopen( file, "rb" );
	fseek( src, 15, SEEK_SET );
	fread( & result[ 0 ], 1, KINECT_RES * 3, src );
	fclose( src );

	return result;
}

void write_ppm( std::vector< unsigned char > const & img, char const * file )
{
	char header[] = "P6 640 480 255 ";

	FILE * dst = fopen( file, "wb" );
	fwrite( header, 1, sizeof( header ) - 1, dst );
	fwrite( & img[ 0 ], 1, 3 * KINECT_RES, dst );
	fclose( dst );
}

unsigned depth2mm( unsigned short kinect_depth )
{
	kinect_depth >>= 3;
	return ( kinect_depth >= 400 && kinect_depth <= 8000 ) ? kinect_depth : 0;
}

void kinect2depth( std::vector< unsigned short > & kinect_frame )
{
	for( int i = 0; i < KINECT_RES; i++ )
		kinect_frame[ i ] = depth2mm( kinect_frame[ i ] );
}

void convert
( 
	char const * src_fmt, 
	char const * dst_fmt, 
	int start, int end,
	std::function< void ( std::vector< unsigned short > const &, std::vector< unsigned char > & ) > depth2rgb
)
{
	for( int i = start; i <= end; i++ )
	{
		char src[ 255 ], dst[ 255 ];
		sprintf( src, src_fmt, i );
		sprintf( dst, dst_fmt, i );

		std::vector< unsigned short > src_img = read_pgm( src );
		std::vector< unsigned char > dst_img( KINECT_RES * 3 );

		kinect2depth( src_img );
		depth2rgb( src_img, dst_img );
		write_ppm( dst_img, dst );

		printf( "\r%d%% done.", i * 100 / 100 );
	}
}

void error
( 
	char const * ref_fmt, 
	char const * enc_fmt,
	char const * csv,
	int start, int end
)
{
	std::map< int, int > diff2count;

	for( int i = start; i <= end; i++ )
	{
		char ref[ 255 ], enc[ 255 ];
		sprintf( ref, ref_fmt, i );
		sprintf( enc, enc_fmt, i );

		std::vector< unsigned char > ref_img = read_ppm( ref );
		std::vector< unsigned char > enc_img = read_ppm( enc );

		for( int j = 0; j < KINECT_RES * 3; j += 3 )
		{
			int a = ref_img[ j ];
			int b = enc_img[ j ];

			diff2count[ abs( a - b ) ]++;
		}

		printf( "\r%d%% done.", i * 100 / 100 );
	}

	FILE * report = fopen( csv, "w" );

	fprintf( report, "Error,\"No. of Occurences\"\n" );

	for( auto it = diff2count.cbegin(); it != diff2count.cend(); it++ )
	{
		fprintf( report, "%d,%d\n", it->first, it->second );
	}

	fclose( report );
}

double linearize( double x )
{
	double x2 = x * x;
	double x3 = x * x2;

	return -3.69889 + 7.96903 * x - 0.0217415 * x2 + 0.0000170182 * x3;
	// (7.15958*10^6 + 568086. x - 408.449 x^2)/(636764. - 594.599 x)
}

int main()
{
	//FILE * shahram = std::fopen( "I:/tmp/rawDepth", "rb" );

	std::vector< float > buffer( 640 * 480 );

	FILE * stream = std::fopen( "I:/tmp/bedroom.depth", "wb" );
	std::fprintf( stream, "KPPL raw depth\n" );

	int version = 2;
	int frameWidth = 640;
	int frameHeight = 480;
	int texelType = 1;
	int nFrames = 100;

	std::fwrite( & version, 4, 1, stream );
	std::fwrite( & frameWidth, 4, 1, stream );
	std::fwrite( & frameHeight, 4, 1, stream );
	std::fwrite( & texelType, 4, 1, stream );
	std::fwrite( & nFrames, 4, 1, stream );

	float matrix[ 16 ];

	for( int i = 1; i <= nFrames; i++ )
	{
		//std::fread( buffer.data(), 4, KINECT_RES, shahram );
		char fileName[ 64 ];
		std::sprintf( fileName, "I:/tmp/bedroom/depth%04d.pgm", i );
		auto depth = read_pgm( fileName );

		for( int i = 0; i < KINECT_RES; i++ )
		{
			unsigned lsb = depth[ i ] & 0xff;
			unsigned msb = depth[ i ] >> 8;

			unsigned raw = ((lsb << 8) | msb) >> 5;
			buffer[ i ] = (float) (100/(-0.00307 * raw + 3.33)) * 0.01f;
			//buffer[ i ] = raw * 0.001f;
		}

		std::fwrite( matrix, 4, 16, stream );
		std::fwrite( buffer.data(), 4, KINECT_RES, stream );
	}

	//std::fclose( shahram );
	std::fclose( stream );

	return 0;
}