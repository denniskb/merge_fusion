#pragma once

#include <reference/dxmath.h>

#include "vector_functions.h"
#include "vector_types_ext.h"



inline float4x4 make_float4x4( svc::float4x4 const & m )
{
	float4x4 result;

	result.col0 = make_float4( m._11, m._21, m._31, m._41 );
	result.col1 = make_float4( m._12, m._22, m._32, m._42 );
	result.col2 = make_float4( m._13, m._23, m._33, m._43 );
	result.col3 = make_float4( m._14, m._24, m._34, m._44 );

	return result;
}