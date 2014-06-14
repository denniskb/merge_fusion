#pragma once

#include <DirectXMath.h>



namespace kifi {
namespace util {

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

}} // namespace

inline kifi::util::float4 operator+( kifi::util::float4 const & a, kifi::util::float4 const & b );
inline kifi::util::float4 operator-( kifi::util::float4 const & a, kifi::util::float4 const & b );
inline kifi::util::float4 operator*( kifi::util::float4 const & a, kifi::util::float4 const & b );
inline kifi::util::float4 operator/( kifi::util::float4 const & a, kifi::util::float4 const & b );

inline kifi::util::vec operator+( kifi::util::vec a, kifi::util::vec b );
inline kifi::util::vec operator-( kifi::util::vec a, kifi::util::vec b );
inline kifi::util::vec operator*( kifi::util::vec a, kifi::util::vec b );
inline kifi::util::vec operator/( kifi::util::vec a, kifi::util::vec b );

inline kifi::util::vec operator*( kifi::util::vec v, kifi::util::mat m );



#pragma region Implementation

#include <algorithm>



namespace kifi {
namespace util {

vec load( float4 const & v )
{
	return DirectX::XMLoadFloat4A( & v );
}

mat load( float4x4 const & m )
{
	return DirectX::XMLoadFloat4x4A( & m );
}

float4 store( vec v )
{
	float4 result;
	DirectX::XMStoreFloat4A( & result, v );
	return result;
}

float4x4 store( mat m )
{
	float4x4 result;
	DirectX::XMStoreFloat4x4A( & result, m );
	return result;
}



float4 make_float4( float s )
{
	return DirectX::XMFLOAT4A( s, s, s, s );
}

vec set( float x, float y, float z, float w )
{
	return DirectX::XMVectorSet( x, y, z, w );
}



template< typename T >
T clamp( T x, T a, T b )
{
	return std::max( a, std::min( x, b ) );
}

float dot( float4 const & a, float4 const & b )
{
	return
		a.x * b.x +
		a.y * b.y +
		a.z * b.z +
		a.w * b.w;
}

vec homogenize( vec v )
{
	return DirectX::XMVectorDivide( v, DirectX::XMVectorPermute< 3, 3, 3, 3 >( v, v ) );
}

float lerp( float a, float b, float weightA, float weightB )
{
	return a * weightA + b * weightB / ( weightA + weightB );
}

bool powerOf2( int x )
{
	return x > 0 && ! (x & (x - 1));
}



unsigned packX( unsigned x )
{
	return x;
}

unsigned packY( unsigned y )
{
	return y << 10;
}

unsigned packZ( unsigned z )
{
	return z << 20;
}

unsigned packInts( unsigned x, unsigned y, unsigned z )
{
	return packX( x ) | packY( y ) | packZ( z );
}



unsigned unpackX( unsigned packedInt )
{
	return packedInt & 0x3ff;
}

unsigned unpackY( unsigned packedInt )
{
	return ( packedInt >> 10 ) & 0x3ff;
}

unsigned unpackZ( unsigned packedInt )
{
	return packedInt >> 20;
}

void unpackInts( unsigned packedInt, unsigned & outX, unsigned & outY, unsigned & outZ )
{
	outX = unpackX( packedInt );
	outY = unpackY( packedInt );
	outZ = unpackZ( packedInt );
}

}} // namespace



kifi::util::float4 operator+( kifi::util::float4 const & a, kifi::util::float4 const & b )
{
	return kifi::util::float4
	(
		a.x + b.x,
		a.y + b.y,
		a.z + b.z,
		a.w + b.w
	);
}

kifi::util::float4 operator-( kifi::util::float4 const & a, kifi::util::float4 const & b )
{
	return kifi::util::float4
	(
		a.x - b.x,
		a.y - b.y,
		a.z - b.z,
		a.w - b.w
	);
}

kifi::util::float4 operator*( kifi::util::float4 const & a, kifi::util::float4 const & b )
{
	return kifi::util::float4
	(
		a.x * b.x,
		a.y * b.y,
		a.z * b.z,
		a.w * b.w
	);
}

kifi::util::float4 operator/( kifi::util::float4 const & a, kifi::util::float4 const & b )
{
	return kifi::util::float4
	(
		a.x / b.x,
		a.y / b.y,
		a.z / b.z,
		0.0f
	);
}



kifi::util::vec operator+( kifi::util::vec a, kifi::util::vec b )
{
	return DirectX::XMVectorAdd( a, b );
}

kifi::util::vec operator-( kifi::util::vec a, kifi::util::vec b )
{
	return DirectX::XMVectorSubtract( a, b );
}

kifi::util::vec operator*( kifi::util::vec a, kifi::util::vec b )
{
	return DirectX::XMVectorMultiply( a, b );
}

kifi::util::vec operator/( kifi::util::vec a, kifi::util::vec b )
{
	return DirectX::XMVectorDivide( a, b );
}



kifi::util::vec operator*( kifi::util::vec v, kifi::util::mat m )
{
	return DirectX::XMVector4Transform( v, m );
}

#pragma endregion