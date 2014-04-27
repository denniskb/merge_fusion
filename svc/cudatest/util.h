#pragma once

#include <DirectXmath.h>

#include <reference/dxmath.h>



inline void ComputeMatrices
(
	svc::float4x4 const & view,
	svc::float4 & outEye,
	svc::float4 & outForward,
	svc::float4x4 & outViewProj,
	svc::float4x4 & outViewToWorld
)
{
	svc::mat _view = svc::load( view );
	svc::mat _viewInv = XMMatrixInverse( nullptr, _view );
	svc::vec _eye = svc::set( 0.0f, 0.0f, 0.0f, 1.0f ) * _viewInv;
	svc::vec _forward = svc::set( 0.0f, 0.0f, -1.0f, 0.0f ) * _viewInv;

	svc::mat _viewProj = _view * DirectX::XMMatrixPerspectiveFovRH( 0.778633444f, 4.0f / 3.0f, 0.8f, 4.0f );

	outEye = svc::store( _eye );
	outForward = svc::store( _forward );
	outViewProj = svc::store( _viewProj );
	outViewToWorld = svc::store( _viewInv );
}