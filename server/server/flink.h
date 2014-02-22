/*
Convenience wrapper around DirectXMath to save typing and
implement some missing functions.
*/

#pragma once

#include <cassert>

#include <DirectXMath.h>

using namespace DirectX;



namespace flink {

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

// extra functions

inline vector homogenize( vector v )
{
	return v / XMVectorPermute< 3, 3, 3, 3 >( v, v );
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

}

// arithmetic operators

inline flink::vector operator*( flink::vector v, flink::matrix m )
{
	return XMVector4Transform( v, m );
}

inline flink::float4 operator-( flink::float4 const & a, flink::float4 const & b )
{
	return flink::float4
	(
		a.x - b.x,
		a.y - b.y,
		a.z - b.z,
		a.w - b.w
	);
}