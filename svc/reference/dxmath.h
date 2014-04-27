/*
Extends DirectXMath with helpers, shortcuts and convenience functions.
*/

#pragma once

#include <DirectXmath.h>



namespace svc {

typedef DirectX::XMFLOAT4A float4;
typedef DirectX::XMFLOAT4X4A float4x4;
typedef DirectX::XMUINT4 uint4;

typedef DirectX::XMVECTOR vec;
typedef DirectX::XMMATRIX mat;

inline vec load( float4 const & v );
inline mat load( float4x4 const & m );
inline float4 store( vec v );
inline float4x4 store( mat m );

inline float4 make_float4( float s );
inline vec set( float x, float y, float z, float w );

template< typename T >
inline T clamp( T x, T a, T b );
inline float dot( float4 const & a, float4 const & b );
inline vec homogenize( vec v );
inline float lerp( float a, float b, float weightA, float weightB );
inline bool powerOf2( int x );

inline unsigned packX( unsigned x );
inline unsigned packY( unsigned y );
inline unsigned packZ( unsigned z );
inline unsigned packInts( unsigned x, unsigned y, unsigned z );

inline unsigned unpackX( unsigned packedInt );
inline unsigned unpackY( unsigned packedInt );
inline unsigned unpackZ( unsigned packedInt );
inline void unpackInts( unsigned packedInt, unsigned & outX, unsigned & outY, unsigned & outZ );

}

inline svc::float4 operator+( svc::float4 const & a, svc::float4 const & b );
inline svc::float4 operator-( svc::float4 const & a, svc::float4 const & b );
inline svc::float4 operator*( svc::float4 const & a, svc::float4 const & b );
inline svc::float4 operator/( svc::float4 const & a, svc::float4 const & b );

inline svc::vec operator+( svc::vec a, svc::vec b );
inline svc::vec operator-( svc::vec a, svc::vec b );
inline svc::vec operator*( svc::vec a, svc::vec b );
inline svc::vec operator/( svc::vec a, svc::vec b );

inline svc::vec operator*( svc::vec v, svc::mat m );



#include "dxmath.inl"