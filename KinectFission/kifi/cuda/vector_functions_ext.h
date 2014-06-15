#pragma once

#include <vector_functions.h>

#include <kifi/cuda/vector_types_ext.h>

#include <kifi/util/math.h>



inline float4x4 make_float4x4( kifi::util::float4x4 const & m )
{
	float4x4 result;

	result.col0 = make_float4( m.row0.x, m.row0.y, m.row0.z, m.row0.w );
	result.col1 = make_float4( m.row1.x, m.row1.y, m.row1.z, m.row1.w );
	result.col2 = make_float4( m.row2.x, m.row2.y, m.row2.z, m.row2.w );
	result.col3 = make_float4( m.row3.x, m.row3.y, m.row3.z, m.row3.w );

	return result;
}