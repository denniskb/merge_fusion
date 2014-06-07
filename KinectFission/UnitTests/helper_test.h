#pragma once

#include <kifi/util/DirectXMathExt.h>



inline void ComputeMatrices
(
	kifi::util::float4x4 const & view,
	kifi::util::float4 & outEye,
	kifi::util::float4 & outForward,
	kifi::util::float4x4 & outViewProj,
	kifi::util::float4x4 & outViewToWorld
)
{
	kifi::util::mat _view = kifi::util::load( view );
	kifi::util::mat _viewInv = XMMatrixInverse( nullptr, _view );
	kifi::util::vec _eye = kifi::util::set( 0.0f, 0.0f, 0.0f, 1.0f ) * _viewInv;
	kifi::util::vec _forward = kifi::util::set( 0.0f, 0.0f, -1.0f, 0.0f ) * _viewInv;

	kifi::util::mat _viewProj = _view * DirectX::XMMatrixPerspectiveFovRH( 0.778633444f, 4.0f / 3.0f, 0.8f, 4.0f );

	outEye = kifi::util::store( _eye );
	outForward = kifi::util::store( _forward );
	outViewProj = kifi::util::store( _viewProj );
	outViewToWorld = kifi::util::store( _viewInv );
}