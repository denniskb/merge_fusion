#pragma once

namespace kppl {

long long fsize( char const * fileName );

inline unsigned packInts( unsigned x, unsigned y, unsigned z )
{
	return z << 20 | y << 10 | x;
}

inline void unpackInts( unsigned packedInt, unsigned & outX, unsigned & outY, unsigned & outZ )
{
	outZ = packedInt >> 20;
	outY = ( packedInt >> 10 ) & 0x3ff;
	outX = packedInt & 0x3ff;
}

}