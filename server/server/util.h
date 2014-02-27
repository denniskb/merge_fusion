#pragma once

namespace kppl {

long long fsize( char const * fileName );

template< unsigned bitsPerComponent >
inline unsigned packInts( unsigned x, unsigned y, unsigned z )
{
	return z << 2 * bitsPerComponent | y << bitsPerComponent | x;
}

template< unsigned bitsPerComponent >
inline void unpackInts( unsigned packedInt, unsigned & outX, unsigned & outY, unsigned & outZ )
{
	outZ = packedInt >> ( 2 * bitsPerComponent );
	outY = ( packedInt >> bitsPerComponent ) & ( ( 1u << bitsPerComponent) - 1 );
	outX = packedInt & ( ( 1u << bitsPerComponent) - 1 );
}

}