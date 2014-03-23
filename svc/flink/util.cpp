#include "util.h"

#include <stdio.h>

#include "vector.h"



long long flink::fsize( char const * fileName )
{
	FILE * test;
	fopen_s( & test, fileName, "rb" );

	_fseeki64( test, 0, SEEK_END );
	long long result = _ftelli64( test );

	fclose( test );

	return result;
}



unsigned flink::packInts( unsigned x, unsigned y, unsigned z )
{
	return z << 20 | y << 10 | x;
}

void flink::unpackInts( unsigned packedInt, unsigned & outX, unsigned & outY, unsigned & outZ )
{
	outZ = unpackZ( packedInt );
	outY = unpackY( packedInt );
	outX = unpackX( packedInt );
}

unsigned flink::unpackX( unsigned packedInt )
{
	return packedInt & 0x3ff;
}

unsigned flink::unpackY( unsigned packedInt )
{
	return ( packedInt >> 10 ) & 0x3ff;
}

unsigned flink::unpackZ( unsigned packedInt )
{
	return packedInt >> 20;
}



bool flink::powerOf2( int x )
{
	return x > 0 && ! ( x & ( x - 1 ) );
} 