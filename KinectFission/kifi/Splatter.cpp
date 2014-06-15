#include <cassert>

#include <kifi/util/math.h>
#include <kifi/util/stop_watch.h>

#include <kifi/Splatter.h>
#include <kifi/Volume.h>



namespace kifi {

// static
void Splatter::Splat( Volume const & volume, std::vector< util::float4 > & outVertices )
{
	assert( 1 == util::pack( 1, 0, 0 ) );

	outVertices.reserve( 1 << 16 );
	outVertices.clear();

	util::chrono::stop_watch t;

	// self, right, top, front
	unsigned deltas[] = { 
		0, 
		util::pack( 1, 0, 0 ),
		util::pack( 0, 1, 0 ),
		util::pack( 0, 0, 1 )
	};

	util::flat_map< unsigned, Voxel >::const_key_iterator voxels[ 4 ];
	std::fill( voxels, voxels + 4, volume.Data().keys_cbegin() );

	auto last = volume.Data().keys_cend() - 1;
	auto values = volume.Data().values_cbegin();

	for( ; voxels[ 0 ] != volume.Data().keys_cend(); ++voxels[ 0 ] )
	{
		Voxel self  = values[ std::distance( volume.Data().keys_cbegin(), voxels[ 0 ] ) ];

		if( 0 == self.Weight() )
			continue;

		voxels[ 1 ] = std::min( voxels[ 0 ] + 1, last );

		for( int i = 2; i <= 3; ++i )
			while( voxels[ i ] < last && * voxels[ i ] < * voxels[ 0 ] + deltas[ i ] )
				++voxels[ i ];

		Voxel right = 
			( * voxels[ 1 ] == * voxels[ 0 ] + deltas[ 1 ] ) ? 
			values[ std::distance( volume.Data().keys_cbegin(), voxels[ 1 ] ) ] : Voxel();

		Voxel top = 
			( * voxels[ 2 ] == * voxels[ 0 ] + deltas[ 2 ] ) ? 
			values[ std::distance( volume.Data().keys_cbegin(), voxels[ 2 ] ) ] : Voxel();

		Voxel front = 
			( * voxels[ 3 ] == * voxels[ 0 ] + deltas[ 3 ] ) ? 
			values[ std::distance( volume.Data().keys_cbegin(), voxels[ 3 ] ) ] : Voxel();

		unsigned x, y, z;
		util::unpack( * voxels[ 0 ], x, y, z );

		util::float4 vert000 = volume.VoxelCenter( util::float4( (float)x, (float)y, (float)z, 0.0f ) );

		float dself, dright, dtop, dfront;
		dself  = self. Distance();
		dright = right.Distance();
		dtop   = top.  Distance();
		dfront = front.Distance();

		if( right.Weight() > 0 && dself * dright < 0.0f )
		{
			util::float4 vert = vert000;
			vert.x += util::lerp( 0.0f, volume.VoxelLength(), abs(dself) / (abs(dself) + abs(dright)) );

			outVertices.push_back( vert );
		}
				
		if( top.Weight() > 0 && dself * dtop < 0.0f )
		{
			util::float4 vert = vert000;
			vert.y += util::lerp( 0.0f, volume.VoxelLength(), abs(dself) / (abs(dself) + abs(dtop)) );

			outVertices.push_back( vert );
		}
				
		if( front.Weight() > 0 && dself * dfront < 0.0f )
		{
			util::float4 vert = vert000;
			vert.z += util::lerp( 0.0f, volume.VoxelLength(), abs(dself) / (abs(dself) + abs(dfront)) );

			outVertices.push_back( vert );
		}
	}

	t.take_time( "tsplat" );
	//t.print();
}

} // namespace