#pragma once

#include <DirectXMath.h>



namespace dlh {

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

inline dlh::float4 operator+( dlh::float4 const & a, dlh::float4 const & b );
inline dlh::float4 operator-( dlh::float4 const & a, dlh::float4 const & b );
inline dlh::float4 operator*( dlh::float4 const & a, dlh::float4 const & b );
inline dlh::float4 operator/( dlh::float4 const & a, dlh::float4 const & b );

inline dlh::vec operator+( dlh::vec a, dlh::vec b );
inline dlh::vec operator-( dlh::vec a, dlh::vec b );
inline dlh::vec operator*( dlh::vec a, dlh::vec b );
inline dlh::vec operator/( dlh::vec a, dlh::vec b );

inline dlh::vec operator*( dlh::vec v, dlh::mat m );



#include "DirectXmathExt.inl"