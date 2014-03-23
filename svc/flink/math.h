/*
Extends DirectXMath with helpers, shortcuts and convenience functions.
*/

#pragma once

#include <algorithm>
#include <cassert>

#include <DirectXMath.h>



namespace flink {

typedef DirectX::XMFLOAT4A float4;
typedef DirectX::XMFLOAT4X4A float4x4;

typedef DirectX::XMVECTOR vec;
typedef DirectX::XMMATRIX mat;

#pragma region float4

inline float4 make_float4( float s )
{
	return float4( s, s, s, s );
}

#pragma endregion

#pragma region vector

inline vec set( float x, float y, float z, float w )
{
	return DirectX::XMVectorSet( x, y, z, w );
}

inline vec load( float4 const & v )
{
	return DirectX::XMLoadFloat4A( & v );
}

inline float4 store( vec v )
{
	float4 result;
	DirectX::XMStoreFloat4A( & result, v );
	return result;
}

#pragma endregion

#pragma region matrix

inline mat load( float4x4 const & m )
{
	return DirectX::XMLoadFloat4x4A( & m );	
}

inline float4x4 store( mat m )
{
	float4x4 result;
	DirectX::XMStoreFloat4x4A( & result, m );
	return result;
}

#pragma endregion

inline float dot( float4 const & a, float4 const & b )
{
	return
		a.x * b.x +
		a.y * b.y +
		a.z * b.z +
		a.w * b.w;
}

inline vec homogenize( vec v )
{
	using DirectX::operator/;
	return v / DirectX::XMVectorPermute< 3, 3, 3, 3 >( v, v );
}

inline float lerp( float a, float b, float weightA, float weightB )
{
	assert( weightA + weightB != 0.0f );

	return a * weightA + b * weightB / ( weightA + weightB );
}

template< typename T >
inline T clamp( T x, T a, T b )
{
	return std::max( a, std::min( x, b ) );
}

}

#pragma region float4 ops

inline flink::float4 operator+( flink::float4 const & a, flink::float4 const & b )
{
	assert( a.w + b.w <= 1.0f );

	return flink::float4
	(
		a.x + b.x,
		a.y + b.y,
		a.z + b.z,
		a.w + b.w
	);
}

inline flink::float4 operator-( flink::float4 const & a, flink::float4 const & b )
{
	assert( a.w - b.w >= 0.0f );

	return flink::float4
	(
		a.x - b.x,
		a.y - b.y,
		a.z - b.z,
		a.w - b.w
	);
}

inline flink::float4 operator*( flink::float4 const & a, flink::float4 const & b )
{
	return flink::float4
	(
		a.x * b.x,
		a.y * b.y,
		a.z * b.z,
		a.w * b.w
	);
}

inline flink::float4 operator/( flink::float4 const & a, flink::float4 const & b )
{
	assert( b.x != 0.0f );
	assert( b.y != 0.0f );
	assert( b.z != 0.0f );
	assert( a.w == 0.0f && b.w == 0.0f );

	return flink::float4
	(
		a.x / b.x,
		a.y / b.y,
		a.z / b.z,
		0.0f
	);
}



inline bool operator<( flink::float4 const & a, flink::float4 const & b )
{
	return
		a.x < b.x &&
		a.y < b.y &&
		a.z < b.z;
}

inline bool operator<=( flink::float4 const & a, flink::float4 const & b )
{
	return
		a.x <= b.x &&
		a.y <= b.y &&
		a.z <= b.z;
}

inline bool operator>( flink::float4 const & a, flink::float4 const & b )
{
	return
		a.x > b.x &&
		a.y > b.y &&
		a.z > b.z;
}

inline bool operator>=( flink::float4 const & a, flink::float4 const & b )
{
	return
		a.x >= b.x &&
		a.y >= b.y &&
		a.z >= b.z;
}

#pragma endregion

#pragma region vector ops

inline flink::vec operator+( flink::vec a, flink::vec b )
{
	return DirectX::XMVectorAdd( a, b );
}

inline flink::vec operator-( flink::vec a, flink::vec b )
{
	return DirectX::XMVectorSubtract( a, b );
}

inline flink::vec operator*( flink::vec a, flink::vec b )
{
	return DirectX::XMVectorMultiply( a, b );
}

inline flink::vec operator/( flink::vec a, flink::vec b )
{
	return DirectX::XMVectorDivide( a, b );
}

inline flink::vec operator*( flink::vec v, flink::mat m )
{
	return DirectX::XMVector4Transform( v, m );
}

#pragma endregion