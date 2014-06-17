#include <algorithm>
#include <iterator>
#include <utility>
#include <vector>

#include <kifi/util/algorithm.h>
#include <kifi/util/math.h>
#include <kifi/util/iterator.h>
#include <kifi/util/vector2d.h>

#include <kifi/Integrator.h>
#include <kifi/Volume.h>
#include <kifi/Voxel.h>

using namespace kifi;
using namespace kifi::util;



namespace {
	struct add_delta
	{
		unsigned delta;
		add_delta( unsigned n ) : delta( n ) {}
		unsigned operator()( unsigned x ) const	{ return x + delta; }
	};
}



// HACK
#include <kifi/util/stop_watch.h>



// static 
void Integrator::Integrate
( 
	Volume & volume,
	vector2d< float > const & frame,

	vec3 eye,
	vec3 forward,

	matrix const & viewProjection,
	matrix4x3 const & viewToWorld
)
{
	m_tmpPointCloud.resize ( frame.size() );
	m_tmpScratchPad.reserve( frame.size() );

	chrono::stop_watch sw;
	size_t nSplats = DepthMap2PointCloud( volume, frame, viewToWorld, m_tmpPointCloud );
	sw.take_time( "tsplat" );

	radix_sort( m_tmpPointCloud.data(), m_tmpPointCloud.data() + nSplats, m_tmpScratchPad.data() );
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
	
	volume.Data().merge_unique(
		m_tmpPointCloud.data(), m_tmpPointCloud.data() + m_tmpPointCloud.size(), Voxel()
	);
	//sw.take_time( "merge" );

	UpdateVoxels( volume, frame, eye, forward, viewProjection );
	//sw.take_time( "tupdate" );

	sw.print_times();
}



// static 
size_t Integrator::DepthMap2PointCloud
(
	Volume const & volume,
	vector2d< float > const & frame,
	matrix4x3 const & viewToWorld,

	std::vector< unsigned > & outPointCloud
)
{
	assert( 0 == frame.width() % 4 );

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

	float4 tmp0 = set( vec3( volume.m_tmpVoxelLenInv ), 0.0f );
	float4 tmp1 = set( vec3( volume.m_tmpNegVoxelLenInvTimesMin - 0.5f ), 0.0f );

	size_t nSplats = 0;

	for( size_t v = 0; v < frame.height(); v++ )
		for( size_t u = 0; u < frame.width(); u += 4 )
		{
			float4 depths = loadu( & frame( u, v ) );

			int depthsValid = any( depths > zero() );
			if( ! depthsValid )
				continue;

			float4 point0 = set( (float) u + 0.0f, - (float) v, -1.0f, 0.0f );
			float4 point1 = set( (float) u + 1.0f, - (float) v, -1.0f, 0.0f );
			float4 point2 = set( (float) u + 2.0f, - (float) v, -1.0f, 0.0f );
			float4 point3 = set( (float) u + 3.0f, - (float) v, -1.0f, 0.0f );

			point0 = point0 * flInv + ppOverFl;
			point1 = point1 * flInv + ppOverFl;
			point2 = point2 * flInv + ppOverFl;
			point3 = point3 * flInv + ppOverFl;
			
			float4 depthx = broadcast< 0 >( depths );
			float4 depthy = broadcast< 1 >( depths );
			float4 depthz = broadcast< 2 >( depths );
			float4 depthw = broadcast< 3 >( depths );

			point0 = point0 * depthx + mask0001;
			point1 = point1 * depthy + mask0001;
			point2 = point2 * depthz + mask0001;
			point3 = point3 * depthw + mask0001;

			point0 *= _viewToWorld;
			point1 *= _viewToWorld;
			point2 *= _viewToWorld;
			point3 *= _viewToWorld;

			point0 = point0 * tmp0 + tmp1;
			point1 = point1 * tmp0 + tmp1;
			point2 = point2 * tmp0 + tmp1;
			point3 = point3 * tmp0 + tmp1;

			int point0Valid = (depthsValid & 0x1) && none( ( point0 < zero() ) | ( point0 >= maxIndex ) );
			int point1Valid = (depthsValid & 0x2) && none( ( point1 < zero() ) | ( point1 >= maxIndex ) );
			int point2Valid = (depthsValid & 0x4) && none( ( point2 < zero() ) | ( point2 >= maxIndex ) );
			int point3Valid = (depthsValid & 0x8) && none( ( point3 < zero() ) | ( point3 >= maxIndex ) );

			float tmp[4];

			if( point0Valid )
			{
				storeu( tmp, point0 );
				outPointCloud[ nSplats++ ] = pack( (uint32_t) tmp[0], (uint32_t) tmp[1], (uint32_t) tmp[2] );
			}

			if( point1Valid )
			{
				storeu( tmp, point1 );
				outPointCloud[ nSplats++ ] = pack( (uint32_t) tmp[0], (uint32_t) tmp[1], (uint32_t) tmp[2] );
			}

			if( point2Valid )
			{
				storeu( tmp, point2 );
				outPointCloud[ nSplats++ ] = pack( (uint32_t) tmp[0], (uint32_t) tmp[1], (uint32_t) tmp[2] );
			}

			if( point3Valid )
			{
				util::storeu( tmp, point3 );
				outPointCloud[ nSplats++ ] = pack( (uint32_t) tmp[0], (uint32_t) tmp[1], (uint32_t) tmp[2] );
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
	ExpandChunksHelper( inOutChunkIndices, pack( 0, 0, 1 ), tmpScratchPad);
	ExpandChunksHelper( inOutChunkIndices, pack( 0, 1, 0 ), tmpScratchPad);
	ExpandChunksHelper( inOutChunkIndices, pack( 1, 0, 0 ), tmpScratchPad);
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
				make_map_iterator( inOutChunkIndices.cbegin(), add_delta( delta ) ), make_map_iterator( inOutChunkIndices.cend(), add_delta( delta ) ),
				tmpScratchPad.data()
			);

			tmpScratchPad.resize( std::distance( tmpScratchPad.data(), tmpNewEnd ) );
			std::swap( tmpScratchPad, inOutChunkIndices );
		}
		break;

	case 1:
		{
			tmpScratchPad.clear();
			unsigned prev = inOutChunkIndices[ 0 ];

			for( std::size_t i = 0; i < inOutChunkIndices.size(); ++i )
			{
				unsigned x = inOutChunkIndices[ i ];

				if( x != prev )
					tmpScratchPad.push_back( prev );
				
				tmpScratchPad.push_back( x );

				prev = x + 1;
			}

			std::swap( tmpScratchPad, inOutChunkIndices );
		}
		break;
	}
}

// static 
void Integrator::UpdateVoxels
(
	Volume & volume,
	vector2d< float > const & frame,

	vec3 eye,
	vec3 forward,
	matrix const & viewProjection
)
{
	float const ndcToUVx = frame.width() * 0.5f;
	float const ndcToUVy = frame.height() * 0.5f;

	auto end = volume.Data().keys_cend();
	for
	( 
		auto it = std::make_pair( volume.Data().keys_cbegin(), volume.Data().values_begin() );
		it.first != end;
		++it.first, ++it.second
	)
	{
		uint32_t x, y, z;
		unpack( * it.first, x, y, z );

		vec3 centerWorld = volume.VoxelCenter( x, y, z );
		float dist = dot( centerWorld - eye, forward );

		vec3 centerNDC = project( centerWorld, viewProjection );
		int u = (int) ( centerNDC.x * ndcToUVx + ndcToUVx );
		int v = (int) ( centerNDC.y * ndcToUVy + ndcToUVy );
		
		if( u < 0 || u >= frame.width() || v < 0 || v >= frame.height() )
			continue;

		float depth = frame( u, frame.height() - v - 1 );
		float signedDist = depth - dist;
				
		int update = ( signedDist >= -volume.TruncationMargin() && 0.0f != depth && dist >= 0.8f );

		it.second->Update( std::min(signedDist, volume.TruncationMargin()), (float)update );
	}
}