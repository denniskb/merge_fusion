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

	DepthSensorParams const & cameraParams,
	std::size_t nPoints
)
{
	util::chrono::stop_watch sw;

	util::float4x4 result = rawEyeToWorldGuess;
	
	for( int i = 0; i < 7; i++ )
		result = AlignStep
		(
			rawDepthMap, result,
			synthPointCloud,
			cameraParams,
			nPoints
		) * result;

	sw.take_time( "ICP (7 iterations)" );
	sw.print_times();

	return result;
}



util::float4x4 ICP::AlignStep
(
	util::vector2d< float > const & rawDepthMap,
	util::float4x4 const & rawEyeToWorldGuess,
		
	std::vector< VertexPositionNormal > const & synthPointCloud,

	DepthSensorParams const & cameraParams,
	std::size_t nPoints
)
{
	using namespace util;

	if( 0 == synthPointCloud.size() || 0 == nPoints )
		return float4x4::identity();

	m_assocs.clear();
	m_assocs.resize( std::min( synthPointCloud.size(), nPoints ) );

	float4x4 srcWorldToClip = cameraParams.EyeToClipRH() * invert_transform( rawEyeToWorldGuess );

	float2 flInv( 1.0f / cameraParams.FocalLengthPixels().x(), 1.0f / cameraParams.FocalLengthPixels().y() );
	float2 ppOverFl = float2
	(
		(0.5f - cameraParams.PrincipalPointPixels().x()),
		(cameraParams.PrincipalPointPixels().y() - 0.5f)
	) * flInv;
	flInv.y() = -flInv.y();

	float halfWidth  = 0.5f * rawDepthMap.width();
	float halfHeight = 0.5f * rawDepthMap.height();

	float3 srcMedianSum( 0.0f );
	float3 dstMedianSum( 0.0f );

	std::size_t step_size = synthPointCloud.size() / nPoints + 1;
	nPoints = 0;
	for( std::size_t i = 0; i < synthPointCloud.size(); i += step_size )
	{
		VertexPositionNormal synth = synthPointCloud[ i ];

		if( isnan( synth.normal ) )
			continue;

		float2 uv = homogenize( srcWorldToClip * float4( synth.position, 1.0f ) ).xy();

		int u = (int) ( uv.x() * halfWidth + halfWidth );
		int v = (int) rawDepthMap.height() - 1 - (int) ( uv.y() * halfHeight + halfHeight );

		if( u < 0 || u >= rawDepthMap.width() ||
			v < 0 || v >= rawDepthMap.height() )
			continue;

		float depth = rawDepthMap( u, v );
		if( 0.0f == depth )
			continue;

		float3 src = ( 
			rawEyeToWorldGuess *
			float4
			(
				( u * flInv.x() + ppOverFl.x() ) * depth,
				( v * flInv.y() + ppOverFl.y() ) * depth,
				-depth, 
				1.0f
			)
		).xyz();

		// refine using compatibility and weighting, also stop on error threshold/count
		// consider stealing FuSci impl (not as expensive as I thought because A can be computed on the fly) // double check that

		float3 diff = src - synth.position;
		if( length_squared( diff ) > 0.01f )
			continue;

		float3 dst = src - dot( diff, synth.normal ) * synth.normal;

		m_assocs[ nPoints ].first  = src;
		m_assocs[ nPoints ].second = dst;
		++nPoints;

		srcMedianSum += src;
		dstMedianSum += dst;
	}
	
	if( 0 == nPoints )
		return float4x4::identity();
	
	float3 srcMedian = (float3) srcMedianSum / (float) nPoints;
	float3 dstMedian = (float3) dstMedianSum / (float) nPoints;

	float
	//kahan_sum< float >
		Sxx( 0.0f ), Sxy( 0.0f ), Sxz( 0.0f ),
		Syx( 0.0f ), Syy( 0.0f ), Syz( 0.0f ),
		Szx( 0.0f ), Szy( 0.0f ), Szz( 0.0f );

	for( std::size_t i = 0; i < nPoints; ++i )
	{
		float3 src = m_assocs[ i ].first;
		float3 dst = m_assocs[ i ].second;

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

	float4x4 N
	( 
		Sxx + Syy + Szz, Syz - Szy      , Szx - Sxz      , Sxy - Syx      ,
		0.0            , Sxx - Syy - Szz, Sxy + Syx      , Szx + Sxz      ,
		0.0            , 0.0            , Syy - Sxx - Szz, Syz + Szy      ,
		0.0            , 0.0            , 0.0            , Szz - Sxx - Syy
	);

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
	float4x4 T = float4x4::identity();
	T.cols[ 3 ] = t;
	T.cols[ 3 ].w() = 1.0f;

	return T * R;
}

} // namespace