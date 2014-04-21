#include "Splatter.h"

#include <cassert>

#include <flink/array_view.h>
#include <flink/util.h>

#include "Volume.h"

#include <flink/timer.h>



// static
void svc::Splatter::Splat( Volume const & volume, std::vector< flink::float4 > & outVertices )
{
	assert( 1 == flink::packX( 1 ) );

	outVertices.reserve( 1 << 16 );
	outVertices.clear();

	flink::timer t;

	Voxel buffer[ 64 ];
	flink::array_view< Voxel, 4, 4, 4 > cache( buffer );

	// self, right, top, front
	unsigned deltas[] = { 0, flink::packX( 1 ), flink::packY( 1 ), flink::packZ( 1 ) };

	flink::flat_map< unsigned, Brick >::const_key_iterator bricks[ 4 ];
	std::fill( bricks, bricks + 4, volume.Data().keys_cbegin() );

	auto last = volume.Data().keys_cend() - 1;
	auto values = volume.Data().values_cbegin();

	for( ; bricks[ 0 ] != volume.Data().keys_cend(); ++bricks[ 0 ] )
	{
		bricks[ 1 ] = std::min( bricks[ 0 ] + 1, last );
		
		// divergence..
		for( int i = 2; i <= 3; ++i )
			while( bricks[ i ] < last && * bricks[ i ] < * bricks[ 0 ] + deltas[ i ] )
				++bricks[ i ];

		for( int i = 0; i < 4; ++i )
		{
			unsigned mask = ( * bricks[ i ] == * bricks[ 0 ] + deltas[ i ] );
			Brick const & b = values[ std::distance( volume.Data().keys_cbegin(), bricks[ i ] ) ];

			unsigned offsetX = ( i == 1 );
			unsigned offsetY = ( i == 2 );
			unsigned offsetZ = ( i == 3 );

			offsetX *= 2;
			offsetY *= 2;
			offsetZ *= 2;

			for( unsigned j = 0; j < 8; ++j )
			{
				unsigned x, y, z;
				Brick::Index1Dto3D( j, x, y, z );

				cache( offsetX + x, offsetY + y, offsetZ + z ) = b[ j ] * mask;
			}
		}

		unsigned bx, by, bz;
		flink::unpackInts( * bricks[ 0 ], bx, by, bz );

		bx *= 2;
		by *= 2;
		bz *= 2;

		for( int i = 0; i < 8; i++ )
		{
			unsigned x, y, z;
			Brick::Index1Dto3D( i, x, y, z );

			Voxel self, right, top, front;

			self  = cache( x + 0, y + 0, z + 0 );
			right = cache( x + 1, y + 0, z + 0 );
			top   = cache( x + 0, y + 1, z + 0 );
			front = cache( x + 0, y + 0, z + 1 );

			if( 0 == self.Weight() )
				continue;

			x += bx;
			y += by;
			z += bz;

			flink::float4 vert000 = volume.VoxelCenter( x, y, z );

			float dself, dright, dtop, dfront;
			dself  = self. Distance( volume.TruncationMargin() );
			dright = right.Distance( volume.TruncationMargin() );
			dtop   = top.  Distance( volume.TruncationMargin() );
			dfront = front.Distance( volume.TruncationMargin() );

			// divergence..

			// TODO: Re-evaluate interpolation (esp. use of weights in lerp)
			if( right.Weight() > 0 && dself * dright < 0.0f )
			{
				flink::float4 vert = vert000;
				vert.x += flink::lerp( 0.0f, volume.VoxelLength(), abs( dright ), abs( dself ) );

				outVertices.push_back( vert );
			}
				
			if( top.Weight() > 0 && dself * dtop < 0.0f )
			{
				flink::float4 vert = vert000;
				vert.y += flink::lerp( 0.0f, volume.VoxelLength(), abs( dtop ), abs( dself ) );

				outVertices.push_back( vert );
			}
				
			if( front.Weight() > 0 && dself * dfront < 0.0f )
			{
				flink::float4 vert = vert000;
				vert.z += flink::lerp( 0.0f, volume.VoxelLength(), abs( dfront ), abs( dself ) );

				outVertices.push_back( vert );
			}
		}
	}

	t.record_time( "tsplat" );
	t.print();
}