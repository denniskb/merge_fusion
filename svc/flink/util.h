#pragma once



namespace flink {

long long fsize( char const * fileName );

unsigned packX( unsigned x );
unsigned packY( unsigned y );
unsigned packZ( unsigned z );

unsigned unpackX( unsigned packedInt );
unsigned unpackY( unsigned packedInt );
unsigned unpackZ( unsigned packedInt );

unsigned packInts( unsigned x, unsigned y, unsigned z );
void unpackInts( unsigned packedInt, unsigned & outX, unsigned & outY, unsigned & outZ );

}