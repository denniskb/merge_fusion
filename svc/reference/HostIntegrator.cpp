#include "HostIntegrator.h"

#include "HostDepthFrame.h"
#include "HostVolume.h"
#include "radix_sort.h"
#include "Timer.h"
#include "util.h"
#include "vector.h"
#include "Voxel.m"



// static 
void svc::HostIntegrator::Integrate
( 
	HostVolume & volume,
	HostDepthFrame const & frame,

	flink::float4 const & eye,
	flink::float4 const & forward,

	flink::float4x4 const & viewProjection,
	flink::float4x4 const & viewToWorld
)
{
	Timer timer;
	SplatBricks( volume, frame, viewToWorld, m_affectedIndices );
	printf( "mark: %fms\n", timer.Time() * 1000.0 );
	timer.Reset();
	radix_sort( m_affectedIndices );
	printf( "sort: %fms\n", timer.Time() * 1000.0 );
	timer.Reset();
	remove_dups( m_affectedIndices );
	printf( "compact: %fms\n", timer.Time() * 1000.0 );
	timer.Reset();

	BricksToVoxels( volume, m_affectedIndices );
	printf( "expand: %fms\n", timer.Time() * 1000.0 );
	timer.Reset();
	
	volume.Indices() = m_affectedIndices;
	// TODO: Encapsulate this functionality !!!
	volume.Voxels().resize( volume.Indices().size() );
	printf( "copy: %fms\n", timer.Time() * 1000.0 );
	timer.Reset();

	UpdateVoxels( volume, frame, eye, forward, viewProjection );
	printf( "integr: %fms\n", timer.Time() * 1000.0 );
	timer.Reset();
}



// static 
void svc::HostIntegrator::SplatBricks
(
	HostVolume const & volume,
	HostDepthFrame const & depthMap,
	flink::float4x4 const & viewToWorld,

	vector< unsigned > & outBrickIndices
)
{
	outBrickIndices.clear();

	flink::matrix _viewToWorld = flink::load( viewToWorld );

	for( int i = 0, res = depthMap.Resolution(); i < res; i++ )
	{
		int y = i / depthMap.Width();
		int x = i % depthMap.Width();

		float depth = depthMap( x, y );
		if( 0.0f == depth )
			continue;

		float xNdc = ( x - 319.5f ) / 319.5f;
		float yNdc = ( 239.5f - y ) / 239.5f;

		flink::float4 pxView
		(
			xNdc * 0.54698249f * depth,
			yNdc * 0.41023687f * depth,
			-depth,
			1.0f
		);

		flink::vector _pxView = flink::load( pxView );
		flink::vector _pxWorld = _pxView * _viewToWorld;

		flink::float4 pxWorld = flink::store( _pxWorld );
		flink::float4 pxVol = volume.BrickIndex( pxWorld );

		if( pxVol < flink::set( 0.5f ) || pxVol >= flink::set( volume.NumBricksInVolume() - 0.5f ) )
			continue;

		pxVol.x -= 0.5f;
		pxVol.y -= 0.5f;
		pxVol.z -= 0.5f;

		unsigned idx = packInts( (unsigned) pxVol.x, (unsigned) pxVol.y, (unsigned) pxVol.z );

		outBrickIndices.push_back( idx );
		outBrickIndices.push_back( idx + 1 );
		outBrickIndices.push_back( idx + 1024 );
		outBrickIndices.push_back( idx + 1025 );
				
		outBrickIndices.push_back( idx + 1048576 );
		outBrickIndices.push_back( idx + 1048577 );
		outBrickIndices.push_back( idx + 1049600 );
		outBrickIndices.push_back( idx + 1049601 );
	}
}

// static 
void svc::HostIntegrator::BricksToVoxels
(
	HostVolume const & volume,
	vector< unsigned > & inOutIndices
)
{
	if( volume.BrickResolution() > 1 )
	{
		Timer timer;

		int const brickSlice = volume.BrickSlice();
		int const brickVolume = volume.BrickVolume();

		int const size = inOutIndices.size();
		inOutIndices.resize( size * brickVolume );

		for( int i = size - 1; i >= 0; i-- )
		{
			unsigned brickX, brickY, brickZ;
			unpackInts( inOutIndices[ i ], brickX, brickY, brickZ );

			for( int j = 0; j < brickVolume; j++ )
			{
				unsigned z = j / brickSlice;
				unsigned y = ( j - z * brickSlice ) / volume.BrickResolution();
				unsigned x = j % volume.BrickResolution();

				inOutIndices[ i * brickVolume + j ] = packInts
				(
					brickX * volume.BrickResolution() + x,
					brickY * volume.BrickResolution() + y,
					brickZ * volume.BrickResolution() + z
				);
			}
		}

		printf( "expand-expand: %fms\n", timer.Time() * 1000.0 );
		timer.Reset();

		// TODO: Sort is overkill. Optimize with specialized permute
		radix_sort( inOutIndices );

		printf( "expand-sort: %fms\n", timer.Time() * 1000.0 );
	}
}

// static 
void svc::HostIntegrator::UpdateVoxels
(
	HostVolume & volume,

	svc::HostDepthFrame const & frame, 

	flink::float4 const & eye,
	flink::float4 const & forward,
	flink::float4x4 const & viewProjection
)
{
	flink::matrix _viewProj = flink::load( viewProjection );
	flink::vector _ndcToUV = flink::set( frame.Width() / 2.0f, frame.Height() / 2.0f, 0, 0 );
		
	for( int i = 0; i < volume.Indices().size(); i++ )
	{
		unsigned x, y, z;
		unpackInts( volume.Indices()[ i ], x, y, z );

		flink::float4 centerWorld = volume.VoxelCenter( x, y, z );
		flink::vector _centerWorld = flink::load( centerWorld );
		
		flink::vector _centerNDC = flink::homogenize( _centerWorld * _viewProj );
		
		flink::vector _centerScreen = _mm_macc_ps( _centerNDC, _ndcToUV, _ndcToUV );
		flink::float4 centerScreen = flink::store( _centerScreen );

		int u = (int) centerScreen.x;
		int v = (int) centerScreen.y;

		float depth = 0.0f;
		// CUDA: Clamp out of bounds access to 0 to avoid divergence
		if( u >= 0 && u < frame.Width() && v >= 0 && v < frame.Height() )
			depth = frame( u, frame.Height() - v - 1 );

		bool update = ( depth != 0.0f );

		float dist = flink::dot( centerWorld - eye, forward );
		float signedDist = depth - dist;
				
		update = update && ( dist >= 0.8f && signedDist >= -volume.TruncationMargin() );

		Voxel vx;
		vx.Update( signedDist, volume.TruncationMargin(), (int) update );
		volume.Voxels()[ i ] = vx;
	}
}