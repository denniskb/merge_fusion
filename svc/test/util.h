#pragma once

#include <DirectXMath.h>

#include <flink/math.h>



inline void ComputeMatrices
(
	flink::float4x4 const & view,
	flink::float4 & outEye,
	flink::float4 & outForward,
	flink::float4x4 & outViewProj,
	flink::float4x4 & outViewToWorld
)
{
	flink::mat _view = flink::load( view );
	flink::mat _viewInv = XMMatrixInverse( nullptr, _view );
	flink::vec _eye = flink::set( 0.0f, 0.0f, 0.0f, 1.0f ) * _viewInv;
	flink::vec _forward = flink::set( 0.0f, 0.0f, -1.0f, 0.0f ) * _viewInv;

	flink::mat _viewProj = _view * DirectX::XMMatrixPerspectiveFovRH( 0.778633444f, 4.0f / 3.0f, 0.8f, 4.0f );

	outEye = flink::store( _eye );
	outForward = flink::store( _forward );
	outViewProj = flink::store( _viewProj );
	outViewToWorld = flink::store( _viewInv );
}