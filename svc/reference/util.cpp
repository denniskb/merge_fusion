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
	outZ = unpackZ( packedInt );
	outY = unpackY( packedInt );
	outX = unpackX( packedInt );
}

unsigned svc::unpackX( unsigned packedInt )
{
	return packedInt & 0x3ff;
}

unsigned svc::unpackY( unsigned packedInt )
{
	return ( packedInt >> 10 ) & 0x3ff;
}

unsigned svc::unpackZ( unsigned packedInt )
{
	return packedInt >> 20;
}



bool svc::powerOf2( int x )
{
	return x > 0 && ! ( x & ( x - 1 ) );
} 



void svc::remove_dups( vector< unsigned > & data )
{
	int i = 0;
	for( int j = 1; j < data.size(); j++ )
		if( data[ i ] != data[ j ] )
			data[ ++i ] = data[ j ];

	data.resize( i + 1 );
}

void svc::remove_value( vector< unsigned > & data, unsigned value )
{
	int i = 0;
	for( int j = 0; j < data.size(); j++ )
		if( data[ j ] != value )
			data[ i++ ] = data[ j ];

	data.resize( i );
}