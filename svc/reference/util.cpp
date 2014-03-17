#include "util.h"

#include <stdio.h>

#include "vector.h"



long long svc::fsize( char const * fileName )
{
	FILE * test;
	fopen_s( & test, fileName, "rb" );

	_fseeki64( test, 0, SEEK_END );
	long long result = _ftelli64( test );

	fclose( test );

	return result;
}



unsigned svc::packInts( unsigned x, unsigned y, unsigned z )
{
	return z << 20 | y << 10 | x;
}

void svc::unpackInts( unsigned packedInt, unsigned & outX, unsigned & outY, unsigned & outZ )
{
	outZ = packedInt >> 20;
	outY = ( packedInt >> 10 ) & 0x3ff;
	outX = packedInt & 0x3ff;
}



bool svc::powerOf2( int x )
{
	return x > 0 && ! ( x & ( x - 1 ) );
} 



void svc::remove_dups( vector< unsigned > & data )
{
	int i = 0;

	for( int j = 1; j < data.size(); j++ )
	{
		unsigned jj = data[ j ];
		if( data[ i ] != jj )
		{
			data[ i + 1 ] = jj;
			i++;
		}
	}

	data.resize( i + 1 );
}