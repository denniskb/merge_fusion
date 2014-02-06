#include <cstdio>
#include <functional>
#include <map>
#include <vector>

static const unsigned KINECT_RES = 640 * 480;

std::vector< unsigned short > read_pgm( char const * file )
{
	std::vector< unsigned short > result( KINECT_RES );

	FILE * src = fopen( file, "rb" );
	fseek( src, 16, SEEK_SET );
	fread( & result[ 0 ], 2, KINECT_RES, src );
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

int main()
{
#if 1
	error
	(
		"C:/Users/admin/Downloads/cafe/cafe_0001a/depth/imrod%03d.ppm",
		"C:/Users/admin/Downloads/cafe/cafe_0001a/depth/imrod1_%03d.ppm",
		"C:/Users/admin/Downloads/cafe/cafe_0001a/depth/imrod1.csv",
		1, 61
	);
#else
	convert
	(
		"C:/Users/admin/Downloads/cafe/cafe_0001a/depth/image%03d.pgm",
		"C:/Users/admin/Downloads/cafe/cafe_0001a/depth/close%03d.ppm",
		1, 100,
		[] ( std::vector< unsigned short > const & src, std::vector< unsigned char > & dst )
		{
			unsigned min = 8000;
			for( int i = 0; i < KINECT_RES; i++ )
			{
				unsigned depth = src[ i ];
				if( depth != 0 && depth < min )
					min = depth;
			}

			for( int i = 0; i < KINECT_RES; i++ )
			{
				unsigned depth = src[ i ];
				if( depth != 0 && depth < min + 2000 )
					depth = ( depth - min ) / ( 2000 / 255.0f ) + 0.5f;
				else
					depth = 0;

				dst[ 3 * i ] = dst[ 3 * i + 1 ] = dst[ 3 * i + 2 ] = depth;
			}
		}
	);

	convert
	(
		"C:/Users/admin/Downloads/cafe/cafe_0001a/depth/image%03d.pgm",
		"C:/Users/admin/Downloads/cafe/cafe_0001a/depth/far%03d.ppm",
		1, 100,
		[] ( std::vector< unsigned short > const & src, std::vector< unsigned char > & dst )
		{
			unsigned min = 8000;
			for( int i = 0; i < KINECT_RES; i++ )
			{
				unsigned depth = src[ i ];
				if( depth != 0 && depth < min )
					min = depth;
			}

			for( int i = 0; i < KINECT_RES; i++ )
			{
				unsigned depth = src[ i ];
				if( depth > min + 6000 )
					depth = min + 6000;

				if( depth != 0 && depth >= min + 2000 )
					depth = ( depth - min - 2000 ) / ( 4000 / 255.0f ) + 0.5f;
				else
					depth = 0;

				dst[ 3 * i ] = dst[ 3 * i + 1 ] = dst[ 3 * i + 2 ] = depth;
			}
		}
	);
#endif

	return 0;
}