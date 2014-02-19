/*
Convenience wrapper around DirectXMath to save typing and
implement some missing functions.
*/

#pragma once

#include <cassert>

#include <DirectXMath.h>



namespace flink {

using namespace DirectX;

// abbreviated type names

typedef XMVECTOR vector;
typedef XMMATRIX matrix;

typedef XMFLOAT4A float4;
typedef XMFLOAT4X4A float4x4;

// set, load and store functions

inline vector set( float x, float y, float z, float w )
{
	return XMVectorSet( x, y, z, w );
}

inline vector load( float4 const * v )
{
	return XMLoadFloat4A( v );
}

inline float4 store( vector v )
{
	float4 result;
	XMStoreFloat4A( & result, v );
	return result;
}

inline matrix load( float4x4 const * m )
{
	return XMLoadFloat4x4A( m );
}

inline float4x4 store( matrix m )
{
	float4x4 result;
	XMStoreFloat4x4A( & result, m );
	return result;
}

// arithmetic operators

inline vector operator*( vector v, matrix m )
{
	return XMVector4Transform( v, m );
}

// extra functions

inline vector homogenize( vector v )
{
	return v / XMVectorPermute< 3, 3, 3, 3 >( v, v );
}

inline float4 operator+( float4 const & a, float4 const & b )
{
	return float4
	(
		a.x + b.x,
		a.y + b.y,
		a.z + b.z,
		a.w + b.w
	);
}

inline float4 operator-( float4 const & a, float4 const & b )
{
	return float4
	(
		a.x - b.x,
		a.y - b.y,
		a.z - b.z,
		a.w - b.w
	);
}

inline float4 operator*( float4 v, float s )
{
	return float4
	(
		v.x * s,
		v.y * s,
		v.z * s,
		v.w * s
	);
}

inline float4 operator/( float4 v, float s )
{
	assert( s != 0.0f );

	float s1 = 1.0f / s;

	return float4
	(
		v.x * s1,
		v.y * s1,
		v.z * s1,
		v.w * s1
	);
}

inline float dot( float4 const & a, float4 const & b )
{
	return
		a.x * b.x + 
		a.y * b.y + 
		a.z * b.z;
}

inline float lerp( float a, float b, float weightA, float weightB )
{
	assert( weightA > 0.0f );
	assert( weightB > 0.0f );

	return ( a * weightA + b * weightB ) / ( weightA + weightB );
}

inline float4 lerp( float4 const & a, float4 const & b, float weightA, float weightB )
{
	assert( weightA > 0.0f );
	assert( weightB > 0.0f );

	return ( a * weightA + b * weightB ) / ( weightA + weightB );
}

}