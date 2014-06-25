#include <algorithm>
#include <iterator>
#include <utility>
#include <vector>

#include <kifi/util/algorithm.h>
#include <kifi/util/functional.h>
#include <kifi/util/math.h>
#include <kifi/util/iterator.h>
#include <kifi/util/vector2d.h>

#include <kifi/Integrator.h>
#include <kifi/Volume.h>
#include <kifi/Voxel.h>

using namespace kifi;



// HACK
#include <kifi/util/stop_watch.h>



// static 
void Integrator::Integrate
( 
	Volume & volume,
	util::vector2d< float > const & frame,

	util::vec3 eye,
	util::vec3 forward,

	util::matrix const & viewProjection,
	util::matrix4x3 const & viewToWorld
)
{
	m_tmpPointCloud.resize ( frame.size() );
	m_tmpScratchPad.reserve( frame.size() );

	util::chrono::stop_watch sw;
	size_t nSplats = DepthMap2PointCloud( volume, frame, viewToWorld, m_tmpPointCloud );
	//sw.take_time( "tsplat" );

	util::radix_sort( m_tmpPointCloud.data(), m_tmpPointCloud.data() + nSplats, m_tmpScratchPad.data() );
	//sw.take_time( "tsort" );

	m_tmpPointCloud.resize(
		std::distance( 
			m_tmpPointCloud.begin(), 
			std::unique( m_tmpPointCloud.begin(), m_tmpPointCloud.begin() + nSplats ) 
		)
	);
	//sw.take_time( "unique" );

	ExpandChunks( m_tmpPointCloud, m_tmpScratchPad );
	//sw.take_time( "expand" );
	
	sw.restart();
	volume.Data().insert(
		m_tmpPointCloud.cbegin(), m_tmpPointCloud.cend(),
		util::make_const_iterator( Voxel() )
	);
	sw.take_time( "merge" );

	//sw.restart();
	UpdateVoxels( volume, frame, eye, forward, viewProjection );
	//sw.take_time( "tupdate" );
	
	sw.print_times();
}



// static 
size_t Integrator::DepthMap2PointCloud
(
	Volume const & volume,
	util::vector2d< float > const & frame,
	util::matrix4x3 const & viewToWorld,

	std::vector< unsigned > & outPointCloud
)
{
	using namespace util;

	assert( 0 == frame.width() % 4 );
	assert( volume.Resolution() > 1 );

	float4x4 _viewToWorld = set( viewToWorld );

	// TODO: Encapsulate in CameraParams struct!
	float4 flInv = set( 1.0f / 585.0f, 1.0f / 585.0f, 1.0f, 1.0f );
	float4 ppOverFl = set
	(
		-((frame.width()  / 2) - 0.5f) / 585.0f, 
		 ((frame.height() / 2) - 0.5f) / 585.0f, 
		0.0f, 
		0.0f
	);
	float4 maxIndex = set( (float) (volume.Resolution() - 1 ) );

	float4 mask0001 = set( 0.0f, 0.0f, 0.0f, 1.0f );
	float4 half = set( 0.5f );

	size_t nSplats = 0;

	for( size_t v = 0; v < frame.height(); v++ )
	{
		float4 point = set( 0.0f, - (float) v, -1.0f, 0.0f );

		for( size_t u = 0; u < frame.width(); u += 4 )
		{
			float4 depths = loadu( & frame( u, v ) );

			int depthsValid = any( depths > zero() );
			if( ! depthsValid )
				continue;

			float4 point0 = loadss( (float) (u + 0) ) + point; 
			float4 point1 = loadss( (float) (u + 1) ) + point;
			float4 point2 = loadss( (float) (u + 2) ) + point;
			float4 point3 = loadss( (float) (u + 3) ) + point;

			point0 = fma( point0, flInv, ppOverFl );
			point1 = fma( point1, flInv, ppOverFl );
			point2 = fma( point2, flInv, ppOverFl );
			point3 = fma( point3, flInv, ppOverFl );
			
			float4 depthx = broadcast< 0 >( depths );
			float4 depthy = broadcast< 1 >( depths );
			float4 depthz = broadcast< 2 >( depths );
			float4 depthw = broadcast< 3 >( depths );

			point0 = fma( point0, depthx, mask0001 );
			point1 = fma( point1, depthy, mask0001 );
			point2 = fma( point2, depthz, mask0001 );
			point3 = fma( point3, depthw, mask0001 );

			point0 *= _viewToWorld;
			point1 *= _viewToWorld;
			point2 *= _viewToWorld;
			point3 *= _viewToWorld;

			point0 = volume.VoxelIndex( point0 ) - half;
			point1 = volume.VoxelIndex( point1 ) - half;
			point2 = volume.VoxelIndex( point2 ) - half;
			point3 = volume.VoxelIndex( point3 ) - half;

			int point0Valid = (depthsValid & 0x1) && none( ( point0 < zero() ) | ( point0 >= maxIndex ) );
			int point1Valid = (depthsValid & 0x2) && none( ( point1 < zero() ) | ( point1 >= maxIndex ) );
			int point2Valid = (depthsValid & 0x4) && none( ( point2 < zero() ) | ( point2 >= maxIndex ) );
			int point3Valid = (depthsValid & 0x8) && none( ( point3 < zero() ) | ( point3 >= maxIndex ) );

			if( point0Valid )
			{
				float tmp[4];
				storeu( tmp, point0 );
				outPointCloud[ nSplats++ ] = pack( (uint32_t) tmp[0], (uint32_t) tmp[1], (uint32_t) tmp[2] );
			}

			if( point1Valid )
			{
				float tmp[4];
				storeu( tmp, point1 );
				outPointCloud[ nSplats++ ] = pack( (uint32_t) tmp[0], (uint32_t) tmp[1], (uint32_t) tmp[2] );
			}

			if( point2Valid )
			{
				float tmp[4];
				storeu( tmp, point2 );
				outPointCloud[ nSplats++ ] = pack( (uint32_t) tmp[0], (uint32_t) tmp[1], (uint32_t) tmp[2] );
			}

			if( point3Valid )
			{
				float tmp[4];
				storeu( tmp, point3 );
				outPointCloud[ nSplats++ ] = pack( (uint32_t) tmp[0], (uint32_t) tmp[1], (uint32_t) tmp[2] );
			}
		}
	}

	return nSplats;
}

// static 
void Integrator::ExpandChunks
( 
	std::vector< unsigned > & inOutChunkIndices,
	std::vector< unsigned > & tmpScratchPad
)
{
	ExpandChunksHelper( inOutChunkIndices, util::pack( 0, 0, 1 ), tmpScratchPad);
	ExpandChunksHelper( inOutChunkIndices, util::pack( 0, 1, 0 ), tmpScratchPad);
	ExpandChunksHelper( inOutChunkIndices, util::pack( 1, 0, 0 ), tmpScratchPad);
}

// static
void Integrator::ExpandChunksHelper
(
	std::vector< unsigned > & inOutChunkIndices,
	unsigned delta,

	std::vector< unsigned > & tmpScratchPad
)
{
	if( 0 == inOutChunkIndices.size() )
		return;

	switch( delta )
	{
	default:
		{
			tmpScratchPad.resize( 2 * inOutChunkIndices.size() );

			auto tmpNewEnd = std::set_union
			(
				inOutChunkIndices.cbegin(), inOutChunkIndices.cend(),
				make_transform_iterator( inOutChunkIndices.cbegin(), util::offset< unsigned >( delta ) ), make_transform_iterator( inOutChunkIndices.cend(), util::offset< unsigned >( delta ) ),
				tmpScratchPad.data()
			);

			tmpScratchPad.resize( std::distance( tmpScratchPad.data(), tmpNewEnd ) );
			std::swap( tmpScratchPad, inOutChunkIndices );
		}
		break;

	case 1:
		{
			tmpScratchPad.resize( inOutChunkIndices.size() * 2 );

			std::size_t dst = 0;
			unsigned prev = inOutChunkIndices[ 0 ];
			for( std::size_t i = 0; i < inOutChunkIndices.size(); ++i )
			{
				unsigned x = inOutChunkIndices[ i ];

				tmpScratchPad[ dst ] = prev;
				dst += ( x != prev );

				tmpScratchPad[ dst++ ] = x;

				prev = x + 1;
			}

			tmpScratchPad.resize( dst );
			std::swap( tmpScratchPad, inOutChunkIndices );
		}
		break;
	}
}

// static 
void Integrator::UpdateVoxels
(
	Volume & volume,
	util::vector2d< float > const & frame,

	util::vec3 eye,
	util::vec3 forward,
	util::matrix const & viewProjection
)
{
	using namespace util;

	float4 ndcToUV = set( frame.width() * 0.5f, frame.height() * 0.5f, 0.0f, 0.0f );

	float4 frameSize = set( (float) frame.width(), (float) frame.height(), std::numeric_limits< float >::max(), std::numeric_limits< float >::max() );

	float4 _eye = set( eye, 1.0f );
	float4 _forward = set( forward, 0.0f );
	float4x4 _viewProjection = set( viewProjection );

	for( size_t i = 0, end = volume.Data().size() / 4 * 4; i < end; i += 4 )
	{
		uint32_t x, y, z;

		unpack( volume.Data().keys_cbegin()[ i + 0 ], x, y, z );
		float4 k0 = set( (float) x, (float) y, (float) z, 1.0f );

		unpack( volume.Data().keys_cbegin()[ i + 1 ], x, y, z );
		float4 k1 = set( (float) x, (float) y, (float) z, 1.0f );

		unpack( volume.Data().keys_cbegin()[ i + 2 ], x, y, z );
		float4 k2 = set( (float) x, (float) y, (float) z, 1.0f );

		unpack( volume.Data().keys_cbegin()[ i + 3 ], x, y, z );
		float4 k3 = set( (float) x, (float) y, (float) z, 1.0f );

		k0 = volume.VoxelCenter( k0 );
		k1 = volume.VoxelCenter( k1 );
		k2 = volume.VoxelCenter( k2 );
		k3 = volume.VoxelCenter( k3 );
		 
		float4 dist0 = dot( k0 - _eye, _forward );
		float4 dist1 = dot( k1 - _eye, _forward );
		float4 dist2 = dot( k2 - _eye, _forward );
		float4 dist3 = dot( k3 - _eye, _forward );

		float dist0f = storess( dist0 );
		float dist1f = storess( dist1 );
		float dist2f = storess( dist2 );
		float dist3f = storess( dist3 );

		k0 = homogenize( k0 * _viewProjection );
		k1 = homogenize( k1 * _viewProjection );
		k2 = homogenize( k2 * _viewProjection );
		k3 = homogenize( k3 * _viewProjection );

		k0 = fma( k0, ndcToUV, ndcToUV );
		k1 = fma( k1, ndcToUV, ndcToUV );
		k2 = fma( k2, ndcToUV, ndcToUV );
		k3 = fma( k3, ndcToUV, ndcToUV );

		int k0valid = dist0f >= 0.8f && none( k0 < zero() | k0 >= frameSize );
		int k1valid = dist1f >= 0.8f && none( k1 < zero() | k1 >= frameSize );
		int k2valid = dist2f >= 0.8f && none( k2 < zero() | k2 >= frameSize );
		int k3valid = dist3f >= 0.8f && none( k3 < zero() | k3 >= frameSize );

		if( k0valid )
		{
			float uv[4];
			storeu( uv, k0 );

			float depth = frame( (unsigned) uv[0], frame.height() - (unsigned) uv[1] - 1 );
			float signedDist = depth - dist0f;
				
			if( signedDist >= -volume.TruncationMargin() && depth > 0.0f )
				volume.Data().values_begin()[ i ].Update( std::min( signedDist, volume.TruncationMargin() ) );
		}

		if( k1valid )
		{
			float uv[4];
			storeu( uv, k1 );

			float depth = frame( (unsigned) uv[0], frame.height() - (unsigned) uv[1] - 1 );
			float signedDist = depth - dist1f;
				
			if( signedDist >= -volume.TruncationMargin() && depth > 0.0f )
				volume.Data().values_begin()[ i + 1 ].Update( std::min( signedDist, volume.TruncationMargin() ) );
		}

		if( k2valid )
		{
			float uv[4];
			storeu( uv, k2 );

			float depth = frame( (unsigned) uv[0], frame.height() - (unsigned) uv[1] - 1 );
			float signedDist = depth - dist2f;
				
			if( signedDist >= -volume.TruncationMargin() && depth > 0.0f )
				volume.Data().values_begin()[ i + 2 ].Update( std::min( signedDist, volume.TruncationMargin() ) );
		}

		if( k3valid )
		{
			float uv[4];
			storeu( uv, k3 );

			float depth = frame( (unsigned) uv[0], frame.height() - (unsigned) uv[1] - 1 );
			float signedDist = depth - dist3f;
				
			if( signedDist >= -volume.TruncationMargin() && depth > 0.0f )
				volume.Data().values_begin()[ i + 3 ].Update( std::min( signedDist, volume.TruncationMargin() ) );
		}	
	}
}