#include <algorithm>
#include <vector>

float lerp( float a, float b, float weightB )
{
	return a + (b - a) * weightB;
}

int clamp( int x, int a, int b )
{
	return std::max( a, std::min( b, x ) );
}

struct image
{
	int width, height;
	std::vector< unsigned char > data;

	float bilinear( float x, float y ) const
	{
		// TODO: Try using euclidean distance

		x *= width;
		y *= height;

		x -= 0.5f;
		y -= 0.5f;

		int ix = (int) x;
		int iy = (int) y;

		x -= ix;
		y -= iy;

		int ix2 = ix + 1;
		int iy2 = iy + 1;

		ix  = clamp( ix , 0, width  - 1 );
		ix2 = clamp( ix2, 0, width  - 1 );
		iy  = clamp( iy , 0, height - 1 );
		iy2 = clamp( iy2, 0, height - 1 );

		x = std::max( 0.0f, x );
		y = std::max( 0.0f, y );

		float tmp0 = lerp( data[ ix + iy  * width ], data[ ix2 + iy  * width ], x );
		float tmp1 = lerp( data[ ix + iy2 * width ], data[ ix2 + iy2 * width ], x );

		return lerp( tmp0, tmp1, y );
	}
};

image read_pgm( char const * file )
{
	image result;

	result.width = 512;
	result.height = 512;

	result.data.resize( result.width * result.height );

	FILE * src = fopen( file, "rb" );
	fseek( src, 38, SEEK_SET );
	fread( result.data.data(), 1, result.data.size(), src );
	fclose( src );

	return result;
}

void write_pgm( image const & img, char const * outFileName )
{
	FILE * dst = fopen( outFileName, "wb" );
	fprintf( dst, "P5 %d %d 255 ", img.width, img.height );
	fwrite( img.data.data(), 1, img.data.size(), dst );
	fclose( dst );
}

image down_sample( image const & img )
{
	image result;

	result.width = img.width / 2;
	result.height = img.height / 2;
	result.data.resize( result.width * result.height );

	for( int y = 0; y < img.height; y += 2 )
		for( int x = 0; x < img.width; x += 2)
		{
			float a, b, c, d;
			a = (float) img.data[ x + 0 + (y + 0) * img.width ];
			b = (float) img.data[ x + 1 + (y + 0) * img.width ];
			c = (float) img.data[ x + 0 + (y + 1) * img.width ];
			d = (float) img.data[ x + 1 + (y + 1) * img.width ];

			int n = (a > 0.0f) + (b > 0.0f) + (c > 0.0f) + (d > 0.0f);
			n = std::max( 1, n );

			result.data[ x/2 + y/2 * result.width ] = (unsigned char) ( ((a + b) + (c + d)) / n );
		}

	return result;
}

void add_noise( image & img )
{
	for( int i = 0; i < img.data.size(); i++ )
		img.data[ i ] = rand() < 10000 ? img.data[ i ] : 0;
}

void fill_holes( image & fine, image const & coarse )
{
	for( int y = 0; y < fine.height; y++ )
		for( int x = 0; x < fine.width; x++ )
		{
			int idx = x + y * fine.width;

			//if( fine.data[ idx ] == 0 )
			//	fine.data[ idx ] = coarse.data[ x/2 + y/2 * coarse.width ];
			if( fine.data[ idx ] == 0 )
				fine.data[ idx ] = (unsigned char) coarse.bilinear( (x + 0.5f) / fine.width, (y + 0.5f) / fine.height );
		}
}

int main()
{
	auto orig = read_pgm( "I:/tmp/orig.pgm" );	

	image mipmaps[ 10 ];
	mipmaps[ 0 ] = read_pgm( "I:/tmp/noise2.pgm" );	;

	//add_noise( mipmaps[ 0 ] );
	//write_pgm( mipmaps[ 0 ], "I:/tmp/test0.pgm" );

	for( int i = 1; i < 10; i++ )
		mipmaps[ i ] = down_sample( mipmaps[ i - 1 ] );

	for( int i = 8; i >= 0; i-- )
		fill_holes( mipmaps[ i ], mipmaps[ i + 1 ] );

	for( int i = 0; i < orig.data.size(); i++ )
		if( orig.data[ i ] < 51 )
			mipmaps[ 0 ].data[ i ] = orig.data[ i ];

	write_pgm( mipmaps[ 0 ], "I:/tmp/recon.pgm" );

	// TODO: Improve reconstr. quality using bilinear interpolation and radius

	return 0;
}