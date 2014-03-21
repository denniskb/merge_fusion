#pragma once

#include "vector.h"



namespace svc {

long long fsize( char const * fileName );

unsigned packInts( unsigned x, unsigned y, unsigned z );
void unpackInts( unsigned packedInt, unsigned & outX, unsigned & outY, unsigned & outZ );

unsigned unpackX( unsigned packedInt );
unsigned unpackY( unsigned packedInt );
unsigned unpackZ( unsigned packedInt );

bool powerOf2( int x );

void remove_dups( vector< unsigned > & data );
void remove_value( vector< unsigned > & data, unsigned value );

}