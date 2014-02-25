#include "DeviceVolume.h"

#include <algorithm>
#include <cassert>

#include "cuda_directives.h"
#include "helper_math_ext.cuh"
#include "HostVolume.h"
#include "KernelDepthFrame.cuh"
#include "KernelVolume.cuh"
#include "Voxel.m"



struct Vertex
{
	unsigned globalIdx;
	float x;
	float y;
	float z;

	kppl_host Vertex() :
		globalIdx( 0 ), x( 0.0f ), y( 0.0f ), z( 0.0f )
	{
	}

	kppl_device Vertex( int globalIdx, float x, float y, float z ) :
		globalIdx( globalIdx ), x( x ), y( y ), z( z )
	{
	}
};

struct VertexCmp
{
	bool operator()( Vertex const & v1, Vertex const & v2 )
	{
		return v1.globalIdx < v2.globalIdx;
	}
} vertexCmp;



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

		kppl::Voxel vx = volume.data[ i ];
		vx.Update( signedDist, volume.truncMargin );
		volume.data[ i ] = vx;
	}
}

// TODO: Extract type (i.e. a queue) for IB/IBptr pair.
__global__ void TriangulateKernel
(
	kppl::KernelVolume volume,
	int const * triTable,
	
	unsigned * IB,
	Vertex * VB,

	unsigned * IBptr,
	unsigned * VBptr
)
{
	int const nThreadsPerDevice = blockDim.x * gridDim.x; // as launched
	int const nVoxels = volume.Volume();

	int const slice = volume.Slice();
	int const resMinus1 = volume.res - 1;

	for( int i = threadIdx.x + blockIdx.x * blockDim.x; i < nVoxels; i += nThreadsPerDevice ) 
	{
		int z = i / slice;
		int y = ( i - z * slice ) / volume.res;
		int x = i % volume.res;

		kppl::Voxel v[ 8 ];
		v[ 2 ] = volume( x, y, z );

		if( 0 == v[ 2 ].Weight() )
			continue;

		int x1 = min( x + 1, resMinus1 );
		int y1 = min( y + 1, resMinus1 );
		int z1 = min( z + 1, resMinus1 );

		v[ 3 ] = volume( x1, y, z );
		v[ 6 ] = volume( x, y1, z );
		v[ 7 ] = volume( x1, y1, z );

		v[ 1 ] = volume( x, y, z1 );
		v[ 0 ] = volume( x1, y, z1 );
		v[ 5 ] = volume( x, y1, z1 );
		v[ 4 ] = volume( x1, y1, z1 );

		// Generate vertices
		float d[ 8 ];
		d[ 1 ] = v[ 1 ].Distance( volume.truncMargin );
		d[ 2 ] = v[ 2 ].Distance( volume.truncMargin );
		d[ 3 ] = v[ 3 ].Distance( volume.truncMargin );
		d[ 6 ] = v[ 6 ].Distance( volume.truncMargin );

		float4 vert000 = volume.VoxelCenter( x, y, z );
		unsigned i000 = volume.Index3Dto1D( x, y, z, volume.res );

		if( v[ 3 ].Weight() > 0 && d[ 2 ] * d[ 3 ] < 0.0f )
		{
			int VBidx = atomicAdd( VBptr, 1 );
			VB[ VBidx ] = Vertex
			(
				3 * i000,
				vert000.x + lerp( 0.0f, volume.vLen, v[ 2 ].Weight() * abs( d[ 3 ] ), v[ 3 ].Weight() * abs( d[ 2 ] ) ),
				vert000.y,
				vert000.z
			);
		}
				
		if( v[ 6 ].Weight() > 0 && d[ 2 ] * d[ 6 ] < 0.0f )
		{
			int VBidx = atomicAdd( VBptr, 1 );
			VB[ VBidx ] = Vertex
			(
				3 * i000 + 1,
				vert000.x,
				vert000.y + lerp( 0.0f, volume.vLen, v[ 2 ].Weight() * abs( d[ 6 ] ), v[ 6 ].Weight() * abs( d[ 2 ] ) ),
				vert000.z
			);
		}
				
		if( v[ 1 ].Weight() > 0 && d[ 2 ] * d[ 1 ] < 0.0f )
		{
			int VBidx = atomicAdd( VBptr, 1 );
			VB[ VBidx ] = Vertex
			(
				3 * i000 + 2,
				vert000.x,
				vert000.y,
				vert000.z + lerp( 0.0f, volume.vLen, v[ 2 ].Weight() * abs( d[ 1 ] ), v[ 1 ].Weight() * abs( d[ 2 ] ) )
			);
		}

		// Generate indices
		bool skip = false;
		for( int i = 0; i < 8; i++ )
			skip |= ( 0 == v[ i ].Weight() );

		if( skip ||					
			x == resMinus1 ||
			y == resMinus1 ||
			z == resMinus1 )
			continue;

		d[ 0 ] = v[ 0 ].Distance( volume.truncMargin );
		d[ 4 ] = v[ 4 ].Distance( volume.truncMargin );
		d[ 5 ] = v[ 5 ].Distance( volume.truncMargin );
		d[ 7 ] = v[ 7 ].Distance( volume.truncMargin );

		int lutIdx = 0;
		for( int i = 0; i < 8; i++ )
			if( d[ i ] < 0 )
				lutIdx |= ( 1u << i );

		// Maps local edge indices to global vertex indices
		unsigned localToGlobal[ 12 ];
		localToGlobal[  0 ] = volume.Index3Dto1D( x, y, z1, volume.res ) * 3;
		localToGlobal[  1 ] = volume.Index3Dto1D( x, y, z, volume.res ) * 3 + 2;
		localToGlobal[  2 ] = volume.Index3Dto1D( x, y, z, volume.res ) * 3;
		localToGlobal[  3 ] = volume.Index3Dto1D( x1, y, z, volume.res ) * 3 + 2;
		localToGlobal[  4 ] = volume.Index3Dto1D( x, y1, z1, volume.res ) * 3;
		localToGlobal[  5 ] = volume.Index3Dto1D( x, y1, z, volume.res ) * 3 + 2;
		localToGlobal[  6 ] = volume.Index3Dto1D( x, y1, z, volume.res ) * 3;
		localToGlobal[  7 ] = volume.Index3Dto1D( x1, y1, z, volume.res ) * 3 + 2;
		localToGlobal[  8 ] = volume.Index3Dto1D( x1, y, z1, volume.res ) * 3 + 1;
		localToGlobal[  9 ] = volume.Index3Dto1D( x, y, z1, volume.res ) * 3 + 1;
		localToGlobal[ 10 ] = volume.Index3Dto1D( x, y, z, volume.res ) * 3 + 1;
		localToGlobal[ 11 ] = volume.Index3Dto1D( x1, y, z, volume.res ) * 3 + 1;

		// reserve memory in IB and write to it
		int IBidx = atomicAdd( IBptr, triTable[ 16 * lutIdx ] );
		for( int i = 0; i < triTable[ 16 * lutIdx ]; i++ )
			IB[ IBidx + i ] = localToGlobal[ triTable[ 16 * lutIdx + i + 1 ] ];
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
		reinterpret_cast< unsigned const * >( rhs.Data() ), 
		reinterpret_cast< unsigned const * >( rhs.Data() + size ),
		m_data.begin()
	);

	return * this;
}

void kppl::DeviceVolume::operator>>( HostVolume & outVolume ) const
{
	assert( outVolume.Resolution() == m_res );

	thrust::copy( m_data.cbegin(), m_data.cend(), reinterpret_cast< unsigned * >( outVolume.Data() ) );
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

void kppl::DeviceVolume::Triangulate( char const * outObj ) const
{
	thrust::device_vector< int > triTable( 256 * 16 );
	thrust::copy( HostVolume::TriTable(), HostVolume::TriTable() + 256 * 16, triTable.begin() );

	// TODO: Dynamically resize vector as required inside kernel
	thrust::device_vector< unsigned > IB( 1000 * 1000 );
	thrust::device_vector< Vertex > VB( 1000 * 1000 );

	thrust::device_vector< unsigned > IBptr( 1 );
	IB[ 0 ] = 0;

	thrust::device_vector< unsigned > VBptr( 1 );
	VBptr[ 0 ] = 0;

	TriangulateKernel<<< 64, 128 >>>
	(
		KernelVolume( const_cast< DeviceVolume & >( * this ) ),
		thrust::raw_pointer_cast( triTable.data() ),
		
		thrust::raw_pointer_cast( IB.data() ),
		thrust::raw_pointer_cast( VB.data() ),

		thrust::raw_pointer_cast( IBptr.data() ),
		thrust::raw_pointer_cast( VBptr.data() )
	);
	
	// TODO: Refactor the rest: Either port it to the GPU, too, or share code between host and device.
	std::vector< Vertex > hVB( VBptr[ 0 ] );
	std::vector< unsigned > hIB( IBptr[ 0 ] );

	thrust::copy( VB.cbegin(), VB.cbegin() + VBptr[ 0 ], hVB.begin() );
	thrust::copy( IB.cbegin(), IB.cbegin() + IBptr[ 0 ], hIB.begin() );

	std::sort( hVB.begin(), hVB.end(), vertexCmp );
	
	Vertex dummy;
	for( int i = 0; i < hIB.size(); i++ )
	{
		dummy.globalIdx = hIB[ i ];
		auto it = std::lower_bound( hVB.cbegin(), hVB.cend(), dummy, vertexCmp );
		hIB[ i ] = (unsigned) ( it - hVB.cbegin() );
	}

	// TODO: Remove unused vertices from VB
	// or test how high their percentage is and possibly leave them in.

	FILE * file;
	fopen_s( & file, outObj, "w" );

	for( int i = 0; i < hVB.size(); i++ )
		fprintf_s( file, "v %f %f %f\n", hVB[ i ].x, hVB[ i ].y, hVB[ i ].z );

	for( int i = 0; i < hIB.size(); i += 3 )
		fprintf_s( file, "f %d %d %d\n", hIB[ i ] + 1, hIB[ i + 1 ] + 1, hIB[ i + 2 ] + 1 );

	fclose( file );
}