#pragma once

#include "vector.h"



namespace flink {

long long fsize( char const * fileName );

unsigned packInts( unsigned x, unsigned y, unsigned z );
void unpackInts( unsigned packedInt, unsigned & outX, unsigned & outY, unsigned & outZ );

unsigned unpackX( unsigned packedInt );
unsigned unpackY( unsigned packedInt );
unsigned unpackZ( unsigned packedInt );

bool powerOf2( int x );

}