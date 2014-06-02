#pragma once

#include <dlh/DirectXMathExt.h>



inline void ComputeMatrices
(
	dlh::float4x4 const & view,
	dlh::float4 & outEye,
	dlh::float4 & outForward,
	dlh::float4x4 & outViewProj,
	dlh::float4x4 & outViewToWorld
)
{
	dlh::mat _view = dlh::load( view );
	dlh::mat _viewInv = XMMatrixInverse( nullptr, _view );
	dlh::vec _eye = dlh::set( 0.0f, 0.0f, 0.0f, 1.0f ) * _viewInv;
	dlh::vec _forward = dlh::set( 0.0f, 0.0f, -1.0f, 0.0f ) * _viewInv;

	dlh::mat _viewProj = _view * DirectX::XMMatrixPerspectiveFovRH( 0.778633444f, 4.0f / 3.0f, 0.8f, 4.0f );

	outEye = dlh::store( _eye );
	outForward = dlh::store( _forward );
	outViewProj = dlh::store( _viewProj );
	outViewToWorld = dlh::store( _viewInv );
}