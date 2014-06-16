#pragma once

#include <kifi/util/math.h>



inline void ComputeMatrices
(
	kifi::util::matrix4x3 const & view,

	kifi::util::vec3 & outEye,
	kifi::util::vec3 & outForward,
	kifi::util::matrix & outViewProj,
	kifi::util::matrix4x3 & outViewToWorld
)
{
	using namespace kifi;

	outViewToWorld = util::invert_transform( view );

	outEye = util::transform_point( util::vec3( 0.0f, 0.0f, 0.0f ), outViewToWorld );
	outForward = util::transform_vector( util::vec3( 0.0f, 0.0f, -1.0f ), outViewToWorld );
	outViewProj = util::matrix( view ) * util::perspective_fov_rh( 0.778633444f, 4.0f / 3.0f, 0.8f, 4.0f );
}