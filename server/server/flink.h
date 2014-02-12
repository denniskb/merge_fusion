/*
Convenience wrapper around DirectXMath to save typing and
implement some missing functions.
*/

#pragma once

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

}