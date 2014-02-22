#include "DeviceVolume.h"

#include <cassert>

#include "helper_math_ext.cuh"
#include "HostVolume.h"
#include "KernelDepthFrame.cuh"
#include "KernelVolume.cuh"
#include "Voxel.m"



__global__ void IntegrateKernel
(
	kppl::KernelVolume volume,
	kppl::KernelDepthFrame const frame,
	float4 const eye,
	float4 const forward,
	float4x4 const viewProjection
)
{
	int const nThreadsPerDevice = blockDim.x * gridDim.x; // as launched
	int const nVoxels = volume.Volume();

	int const slice = volume.Slice();
	float const halfWidth = frame.width / 2;
	float const halfHeight = frame.height / 2;
	
	for( int i = threadIdx.x + blockIdx.x * blockDim.x; i < nVoxels; i += nThreadsPerDevice ) 
	{
		int z = i / slice;
		int y = ( i - z * slice ) / volume.res;
		int x = i % volume.res;

		float4 centerWorld = volume.VoxelCenter( x, y, z );
		float4 centerNDC = homogenize( centerWorld * viewProjection );

		int u = (int) fmaf( centerNDC.x, halfWidth, halfWidth );
		int v = (int) fmaf( centerNDC.y, halfHeight, halfHeight );
		
		// TODO: Test removing early exits for a more homogenous execution path
		if( u < 0 || u >= frame.width || v < 0 || v >= frame.height )
			continue;

		// TODO: Try and use texture object
		float depth = frame( u, frame.height - v - 1 );

		if( depth == 0.0f )
			continue;

		float dist = dot( centerWorld - eye, forward );
		float signedDist = depth - dist;
				
		if( dist < 0.8f || signedDist < -volume.truncMargin )
			continue;

		volume.data[ i ].Update( signedDist, volume.truncMargin, volume.packScale, volume.unpackScale );
	}
}



kppl::DeviceVolume::DeviceVolume( int resolution, float sideLength, float truncationMargin ) :
	m_data( resolution * resolution * resolution ),
	m_res( resolution ), 
	m_sideLen( sideLength ),
	m_truncMargin( truncationMargin ),
	m_nUpdates( 0 )
{
	assert( resolution > 0 );
	assert( sideLength > 0.0f );
	assert( truncationMargin > 0.0f );
}



int kppl::DeviceVolume::Resolution() const
{
	return m_res;
}

float kppl::DeviceVolume::SideLength() const
{
	return m_sideLen;
}

float kppl::DeviceVolume::TruncationMargin() const
{
	return m_truncMargin;
}
	
kppl::Voxel * kppl::DeviceVolume::Data()
{
	return reinterpret_cast< Voxel * >( thrust::raw_pointer_cast( m_data.data() ) );
}

kppl::Voxel const * kppl::DeviceVolume::Data() const
{
	return reinterpret_cast< Voxel const * >( thrust::raw_pointer_cast( m_data.data() ) );
}



kppl::DeviceVolume & kppl::DeviceVolume::operator<<( HostVolume const & rhs )
{
	m_res = rhs.Resolution();
	m_sideLen = rhs.SideLength();
	m_truncMargin = rhs.TrunactionMargin();

	int size = rhs.Resolution() * rhs.Resolution() * rhs.Resolution();
	m_data.resize( size );

	thrust::copy
	(
		reinterpret_cast< short const * >( & rhs( 0, 0, 0 ) ), 
		reinterpret_cast< short const * >( & rhs( 0, 0, 0 ) + size ),
		m_data.begin()
	);

	return * this;
}

void kppl::DeviceVolume::operator>>( HostVolume & outVolume ) const
{
	assert( outVolume.Resolution() == m_res );

	thrust::copy( m_data.cbegin(), m_data.cend(), reinterpret_cast< short * >( & outVolume( 0, 0, 0 ) ) );
}



void kppl::DeviceVolume::Integrate
(
	DeviceDepthFrame const & frame,
	flink::float4 const & eye,
	flink::float4 const & forward,
	flink::float4x4 const & viewProjection
)
{
	assert( m_nUpdates < Voxel::MAX_WEIGHT() );

	IntegrateKernel<<< 64, 128 >>>
	(
		KernelVolume( * this ),
		KernelDepthFrame( frame ),
		make_float4( eye ),
		make_float4( forward ),
		float4x4( viewProjection )
	);

	m_nUpdates++;
}