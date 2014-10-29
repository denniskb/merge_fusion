#include <algorithm>
#include <cassert>
#include <iterator>
#include <utility>
#include <vector>

#include <kifi/util/algorithm.h>
#include <kifi/util/functional.h>
#include <kifi/util/math.h>
#include <kifi/util/iterator.h>
#include <kifi/util/vector2d.h>

#include <kifi/DepthSensorParams.h>
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

	DepthSensorParams const & cameraParams,
	util::float4x4 const & eyeToWorld
)
{
	assert( frame.width() == cameraParams.ResolutionPixels().x() );
	assert( frame.height() == cameraParams.ResolutionPixels().y() );

	util::chrono::stop_watch sw;

	m_tmpPointCloud.resize ( frame.size() );
	m_tmpScratchPad.reserve( frame.size() );

	util::float4x4 worldToClip = cameraParams.EyeToClipRH() * util::invert_transform( eyeToWorld );


	std::size_t nSplats = DepthMap2PointCloud( volume, frame, cameraParams, eyeToWorld, m_tmpPointCloud );
	
	util::radix_sort( m_tmpPointCloud.data(), m_tmpPointCloud.data() + nSplats, m_tmpScratchPad.data() );

	m_tmpPointCloud.resize(
		std::distance( 
			m_tmpPointCloud.begin(), 
			std::unique( m_tmpPointCloud.begin(), m_tmpPointCloud.begin() + nSplats ) 
		)
	);

	// Turn each index in tmpPointCloud into 27 indices -- without duplicates -- to have enough margin for normal computation.
	ExpandChunks( m_tmpPointCloud, m_tmpScratchPad );
	ExpandChunks( m_tmpPointCloud, m_tmpScratchPad );

	volume.Data().insert(
		m_tmpPointCloud.cbegin(), m_tmpPointCloud.cend(),
		util::make_const_iterator( Voxel() )
	);

	UpdateVoxels( volume, frame, worldToClip );
	
	sw.take_time( "Integration" );
	sw.print_times();
}



// static 
std::size_t Integrator::DepthMap2PointCloud
(
	Volume const & volume,
	util::vector2d< float > const & frame,

	DepthSensorParams const & cameraParams,
	util::float4x4 const & eyeToWorld,

	std::vector< unsigned > & outPointCloud
)
{
	using namespace util;

	assert( 0 == frame.width() % 4 );
	assert( volume.Resolution() > 1 );

	matrix _eyeToWorld = load( eyeToWorld );

	vector flInv = set
	(
		1.0f / cameraParams.FocalLengthNorm().x() / cameraParams.ResolutionPixels().x(),
		1.0f / cameraParams.FocalLengthNorm().y() / cameraParams.ResolutionPixels().y(),
		1.0f, 1.0f 
	);

	vector ppOverFl = set
	(
		(0.5f - cameraParams.PrincipalPointNorm().x() * cameraParams.ResolutionPixels().x()),
		(cameraParams.PrincipalPointNorm().y() * cameraParams.ResolutionPixels().y() - 0.5f),
		0.0f, 
		0.0f
	) * flInv;
	vector maxIndex = set( (float) (volume.Resolution() - 2 ) );

	vector mask0001 = set( 0.0f, 0.0f, 0.0f, 1.0f );
	vector half = set( 0.5f );

	std::size_t nSplats = 0;

	for( std::size_t v = 0; v < frame.height(); v++ )
	{
		vector point = set( 0.0f, - (float) v, -1.0f, 0.0f );

		for( std::size_t u = 0; u < frame.width(); u += 4 )
		{
			vector depths = loadu( & frame( u, v ) );

			int depthsValid = any( depths > zero() );
			if( ! depthsValid )
				continue;

			vector point0 = loadss( (float) (u + 0) ) + point; 
			vector point1 = loadss( (float) (u + 1) ) + point;
			vector point2 = loadss( (float) (u + 2) ) + point;
			vector point3 = loadss( (float) (u + 3) ) + point;

			point0 = point0 * flInv + ppOverFl;
			point1 = point1 * flInv + ppOverFl;
			point2 = point2 * flInv + ppOverFl;
			point3 = point3 * flInv + ppOverFl;
			
			vector depthx = broadcast< 0 >( depths );
			vector depthy = broadcast< 1 >( depths );
			vector depthz = broadcast< 2 >( depths );
			vector depthw = broadcast< 3 >( depths );

			point0 = point0 * depthx + mask0001;
			point1 = point1 * depthy + mask0001;
			point2 = point2 * depthz + mask0001;
			point3 = point3 * depthw + mask0001;

			point0 = _eyeToWorld * point0;
			point1 = _eyeToWorld * point1;
			point2 = _eyeToWorld * point2;
			point3 = _eyeToWorld * point3;

			point0 = volume.VoxelIndex( point0 ) - half;
			point1 = volume.VoxelIndex( point1 ) - half;
			point2 = volume.VoxelIndex( point2 ) - half;
			point3 = volume.VoxelIndex( point3 ) - half;

			int point0Valid = (depthsValid & 0x1) && all( point0 >= zero() & point0 < maxIndex );
			int point1Valid = (depthsValid & 0x2) && all( point1 >= zero() & point1 < maxIndex );
			int point2Valid = (depthsValid & 0x4) && all( point2 >= zero() & point2 < maxIndex );
			int point3Valid = (depthsValid & 0x8) && all( point3 >= zero() & point3 < maxIndex );

			if( point0Valid )
			{
				float4 p = store( point0 );
				outPointCloud[ nSplats++ ] = pack( (unsigned) p.x(), (unsigned) p.y(), (unsigned) p.z() );
			}

			if( point1Valid )
			{
				float4 p = store( point1 );
				outPointCloud[ nSplats++ ] = pack( (unsigned) p.x(), (unsigned) p.y(), (unsigned) p.z() );
			}

			if( point2Valid )
			{
				float4 p = store( point2 );
				outPointCloud[ nSplats++ ] = pack( (unsigned) p.x(), (unsigned) p.y(), (unsigned) p.z() );
			}

			if( point3Valid )
			{
				float4 p = store( point3 );
				outPointCloud[ nSplats++ ] = pack( (unsigned) p.x(), (unsigned) p.y(), (unsigned) p.z() );
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
	util::float4x4 const & worldToClip
)
{
	using namespace util;

	vector ndcToUV = set( frame.width() * 0.5f, frame.height() * 0.5f, 0.0f, 0.0f );

	vector minUV = set( 0.0f, 0.0f, -1.0f, -std::numeric_limits< float >::max() );
	vector maxUV = set( (float) frame.width(), (float) frame.height(), 1.0f, std::numeric_limits< float >::max() );

	matrix _worldToClip = load( worldToClip );

	for( std::size_t i = 0, end = volume.Data().size() / 4 * 4; i < end; i += 4 )
	{
		unsigned x, y, z;

		unpack( volume.Data().keys_cbegin()[ i + 0 ], x, y, z );
		vector k0 = set( (float) x, (float) y, (float) z, 1.0f );

		unpack( volume.Data().keys_cbegin()[ i + 1 ], x, y, z );
		vector k1 = set( (float) x, (float) y, (float) z, 1.0f );

		unpack( volume.Data().keys_cbegin()[ i + 2 ], x, y, z );
		vector k2 = set( (float) x, (float) y, (float) z, 1.0f );

		unpack( volume.Data().keys_cbegin()[ i + 3 ], x, y, z );
		vector k3 = set( (float) x, (float) y, (float) z, 1.0f );

		k0 = volume.VoxelCenter( k0 );
		k1 = volume.VoxelCenter( k1 );
		k2 = volume.VoxelCenter( k2 );
		k3 = volume.VoxelCenter( k3 );

		k0 = _worldToClip * k0;
		k1 = _worldToClip * k1;
		k2 = _worldToClip * k2;
		k3 = _worldToClip * k3;

		vector k0w = broadcast< 3 >( k0 );
		vector k1w = broadcast< 3 >( k1 );
		vector k2w = broadcast< 3 >( k2 );
		vector k3w = broadcast< 3 >( k3 );

		float dist0f = storess( k0w );
		float dist1f = storess( k1w );
		float dist2f = storess( k2w );
		float dist3f = storess( k3w );

		k0 /= k0w;
		k1 /= k1w;
		k2 /= k2w;
		k3 /= k3w;

		k0 = k0 * ndcToUV + ndcToUV;
		k1 = k1 * ndcToUV + ndcToUV;
		k2 = k2 * ndcToUV + ndcToUV;
		k3 = k3 * ndcToUV + ndcToUV;
		
		if( all( k0 >= minUV & k0 < maxUV ) )
		{
			float4 uv = store( k0 );

			float depth = frame( (unsigned) uv.x(), frame.height() - (unsigned) uv.y() - 1 );
			float signedDist = depth - dist0f;
				
			if( depth > 0.0f && signedDist >= -volume.TruncationMargin() )
				volume.Data().values_begin()[ i ].Update( std::min( signedDist, volume.TruncationMargin() ) );
		}

		if( all( k1 >= minUV & k1 < maxUV ) )
		{
			float4 uv = store( k1 );

			float depth = frame( (unsigned) uv.x(), frame.height() - (unsigned) uv.y() - 1 );
			float signedDist = depth - dist1f;
				
			if( depth > 0.0f && signedDist >= -volume.TruncationMargin() )
				volume.Data().values_begin()[ i + 1 ].Update( std::min( signedDist, volume.TruncationMargin() ) );
		}

		if( all( k2 >= minUV & k2 < maxUV ) )
		{
			float4 uv = store( k2 );

			float depth = frame( (unsigned) uv.x(), frame.height() - (unsigned) uv.y() - 1 );
			float signedDist = depth - dist2f;
				
			if( depth > 0.0f && signedDist >= -volume.TruncationMargin() )
				volume.Data().values_begin()[ i + 2 ].Update( std::min( signedDist, volume.TruncationMargin() ) );
		}

		if( all( k3 >= minUV & k3 < maxUV ) )
		{
			float4 uv = store( k3 );

			float depth = frame( (unsigned) uv.x(), frame.height() - (unsigned) uv.y() - 1 );
			float signedDist = depth - dist3f;
				
			if( depth > 0.0f && signedDist >= -volume.TruncationMargin() )
				volume.Data().values_begin()[ i + 3 ].Update( std::min( signedDist, volume.TruncationMargin() ) );
		}	
	}
}