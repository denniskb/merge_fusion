#include <kifi/util/numeric.h>

#include <kifi/ICP.h>

// HACK
#include <kifi/util/stop_watch.h>



namespace kifi {

util::float4x4 ICP::Align
(
	util::vector2d< float > const & rawDepthMap,
	util::float4x4 const & rawEyeToWorldGuess,
		
	std::vector< VertexPositionNormal > const & synthPointCloud,

	DepthSensorParams const & cameraParams
)
{
	util::float4x4 result = rawEyeToWorldGuess;
	
	for( int i = 0; i < 7; i++ )
		result = AlignStep
		(
			rawDepthMap, result,
			synthPointCloud,
			cameraParams
		) * result;

	return result;
}



util::float4x4 ICP::AlignStep
(
	util::vector2d< float > const & rawDepthMap,
	util::float4x4 const & rawEyeToWorldGuess,
		
	std::vector< VertexPositionNormal > const & synthPointCloud,

	DepthSensorParams const & cameraParams
)
{
	using namespace util;

	float4x4 srcWorldToClip = cameraParams.EyeToClipRH() * invert_transform( rawEyeToWorldGuess );

	float2 flInv( 1.0f / cameraParams.FocalLengthPixels().x(), 1.0f / cameraParams.FocalLengthPixels().y() );
	float2 ppOverFl = float2
	(
		(0.5f - cameraParams.PrincipalPointPixels().x()),
		(cameraParams.PrincipalPointPixels().y() - 0.5f)
	) * flInv;

	float halfWidth  = 0.5f * rawDepthMap.width();
	float halfHeight = 0.5f * rawDepthMap.height();

	float3 srcMedianSum( 0.0f );
	float3 dstMedianSum( 0.0f );

	m_assocs.clear();
	m_assocs.resize( synthPointCloud.size() );
	unsigned nAssocs = 0;

	chrono::stop_watch sw;

	for( std::size_t i = 0; i < synthPointCloud.size(); i++ )
	{
		auto dst = synthPointCloud[ i ];

		// project dst onto src
		float4 uv = homogenize( srcWorldToClip * float4( dst.position, 1.0f ) );

		// construct 3D point out of uv and raw depth (640x480)
		int u = (int) ( uv.x() * halfWidth + halfWidth );
		int v = (int) rawDepthMap.height() - 1 - (int) ( uv.y() * halfHeight + halfHeight );

		if( u < 0 || u >= rawDepthMap.width() ||
			v < 0 || v >= rawDepthMap.height() )
			continue;

		float depth = rawDepthMap( u, v );
		if( 0.0f == depth ) // invalid depth
			continue;

		float4 raw( (float) u, - (float) v, -depth, 1.0f );

		raw.x() = ( raw.x() * flInv.x() + ppOverFl.x() ) * depth;
		raw.y() = ( raw.y() * flInv.y() + ppOverFl.y() ) * depth;

		float3 src = ( rawEyeToWorldGuess * raw ).xyz();

		// TODO: Make this work with 0 assocs (return identity)
		// refine using compatibility and weighting, also stop on error threshold/count
		// consider stealing FuSci impl (not as expensive as I thought because A can be computed on the fly) // double check that

		// check compatibility
		float3 diff = src - dst.position;
		if( length_squared( diff ) > 0.0025f )
			continue;

		// construct virtual point to reflect point-to-plane distance
		float3 virt = src - dot( diff, dst.normal ) * dst.normal;

		m_assocs[ nAssocs++ ] = std::make_pair( src, virt );

		srcMedianSum += src;
		dstMedianSum += virt;
	}

	sw.take_time( "icp_assoc" );

	if( 0 == nAssocs )
		return float4x4::identity();
	
	float3 srcMedian = srcMedianSum / (float) nAssocs;
	float3 dstMedian = dstMedianSum / (float) nAssocs;

	float
		Sxx = 0.0f, Sxy = 0.0f, Sxz = 0.0f,
		Syx = 0.0f, Syy = 0.0f, Syz = 0.0f,
		Szx = 0.0f, Szy = 0.0f, Szz = 0.0f;

	// associations computed => Horn
	for( std::size_t i = 0; i < nAssocs; ++i )
	{
		// all points are valid and compatible, no-brainer
		auto x = m_assocs[ i ];

		float3 src = x.first;
		float3 dst = x.second;

		src -= srcMedian;
		dst -= dstMedian;

		Sxx += src.x() * dst.x();
        Sxy += src.x() * dst.y();
        Sxz += src.x() * dst.z();

        Syx += src.y() * dst.x();
        Syy += src.y() * dst.y();
        Syz += src.y() * dst.z();

        Szx += src.z() * dst.x();
        Szy += src.z() * dst.y();
        Szz += src.z() * dst.z();
	}

	sw.take_time( "icp_S" );
	sw.print_times();
	
	float4x4 N
	( 
		Sxx + Syy + Szz, Syz - Szy      , Szx - Sxz      , Sxy - Syx      ,
		0.0            , Sxx - Syy - Szz, Sxy + Syx      , Szx + Sxz      ,
		0.0            , 0.0            , Syy - Sxx - Szz, Syz + Szy      ,
		0.0            , 0.0            , 0.0            , Szz - Sxx - Syy
	);

	// N is symmetric by construction.
	N(1, 0) = N(0, 1);
	N(2, 0) = N(0, 2);
	N(3, 0) = N(0, 3);
	N(2, 1) = N(1, 2);
	N(3, 1) = N(1, 3);
	N(3, 2) = N(2, 3);

	float4 ev;
	eigen( N, ev, N );

	int imaxev = 0;
    for( int ii = 1; ii < 4; ii++ )
        if( ev[ ii ] > ev[ imaxev ] )
            imaxev = ii;

	float4 q = normalize( N.cols[ imaxev ] );

	float4x4 R = float4x4::identity();
    float xy = q.x() * q.y();
    float xz = q.x() * q.z();
    float xw = q.x() * q.w();
    float yy = q.y() * q.y();
    float yz = q.y() * q.z();
    float yw = q.y() * q.w();
    float zz = q.z() * q.z();
    float zw = q.z() * q.w();
    float ww = q.w() * q.w();

    R(0, 0) = 1.0f - 2.0f * (zz + ww);
    R(0, 1) = 2.0f * (yz - xw);
    R(0, 2) = 2.0f * (yw + xz);

    R(1, 0) = 2.0f * (yz + xw);
    R(1, 1) = 1.0f - 2.0f * (yy + ww);
    R(1, 2) = 2.0f * (zw - xy);

    R(2, 0) = 2.0f * (yw - xz);
    R(2, 1) = 2.0f * (zw + xy);
    R(2, 2) = 1.0f - 2.0f * (yy + zz);

	float4 t = float4( dstMedian, 1.0f ) - R * float4( srcMedian, 1.0f );

	// translation matrix:
	float4x4 T = float4x4::identity();
	T.cols[ 3 ] = t;
	T.cols[ 3 ].w() = 1.0f;

	float4x4 result = T * R;

	return result;
}

} // namespace