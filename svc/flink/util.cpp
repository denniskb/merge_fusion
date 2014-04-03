#include "util.h"

#include <cstdio>



long long flink::fsize( char const * fileName )
{
	FILE * test;
	fopen_s( & test, fileName, "rb" );

	_fseeki64( test, 0, SEEK_END );
	long long result = _ftelli64( test );

	fclose( test );

	return result;
}



unsigned flink::packX( unsigned x )
{
	return x;
}

unsigned flink::packY( unsigned y )
{
	return y << 10;
}

unsigned flink::packZ( unsigned z )
{
	return z << 20;
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

unsigned flink::packInts( unsigned x, unsigned y, unsigned z )
{
	return packX( x ) | packY( y ) | packZ( z );
}

void flink::unpackInts( unsigned packedInt, unsigned & outX, unsigned & outY, unsigned & outZ )
{
	outZ = unpackZ( packedInt );
	outY = unpackY( packedInt );
	outX = unpackX( packedInt );
}