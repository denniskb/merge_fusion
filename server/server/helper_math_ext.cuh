/*
Extending cuda's helper_math with some useful functions
*/

#pragma once

#include "force_nvcc.h"

#include <vector_types.h>
#include <helper_math.h>

#include "flink.h"



struct float4x4
{
	float4 col0;
	float4 col1;
	float4 col2;
	float4 col3;

	inline __host__ float4x4( flink::float4x4 const & m )
	{
		col0.x = m._11;
		col1.x = m._12;
		col2.x = m._13;
		col3.x = m._14;

		col0.y = m._21;
		col1.y = m._22;
		col2.y = m._23;
		col3.y = m._24;

		col0.z = m._31;
		col1.z = m._32;
		col2.z = m._33;
		col3.z = m._34;

		col0.w = m._41;
		col1.w = m._42;
		col2.w = m._43;
		col3.w = m._44;
	}
};

inline __host__ float4 make_float4( flink::float4 const & v )
{
	return make_float4( v.x, v.y, v.z, v.w );
}

inline __device__ float4 operator*( float4 const & v, float4x4 const & m )
{
	return make_float4
	(
		dot( v, m.col0 ),
		dot( v, m.col1 ),
		dot( v, m.col2 ),
		dot( v, m.col3 )
	);
}

inline __device__ float4 homogenize( float4 const & v )
{
	return v / v.w;
}

inline __device__ float lerp( float a, float b, float weightA, float weightB )
{
	return ( a * weightA + b * weightB ) / ( weightA + weightB );
}