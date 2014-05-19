#pragma once

#include <helper_math.h>
#include <vector_types.h>

#include "vector_types_ext.h"



inline __device__ unsigned horizontal_sum( uint4 v )
{
	return v.x + v.y + v.z + v.w;
}

inline __device__ float4 operator*( float4 v, float4x4 m )
{
	float4 result;

	result.x = dot( v, m.col0 );
	result.y = dot( v, m.col1 );
	result.z = dot( v, m.col2 );
	result.w = dot( v, m.col3 );

	return result;
}

inline __device__ unsigned packInts( unsigned x, unsigned y, unsigned z )
{
	return x | y << 10 | z << 20;
}

template< bool includeSelf >
inline __device__ void scan( uint4 & v );

template<>
inline __device__ void scan< true >( uint4 & v )
{
	v.y += v.x;
	v.z += v.y;
	v.w += v.z;
}

template<>
inline __device__ void scan< false >( uint4 & v )
{
	uint4 tmp = v;
	scan< true >( v );
	v -= tmp;
}