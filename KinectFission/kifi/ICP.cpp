#include <kifi/util/numeric.h>

#include <kifi/ICP.h>



namespace kifi {

util::float4x4 ICP::Align
(
	util::vector2d< float > const & rawDepthMap,
	util::float4x4 const & rawDepthMapEyeToWorldGuess,
		
	util::vector2d< VertexPositionNormal > const & synthDepthBuffer,
	util::float4x4 const & synthDepthBufferEyeToWorld,

	DepthSensorParams const & cameraParams
)
{
	util::float4x4 result = rawDepthMapEyeToWorldGuess;
	
	for( int i = 0; i < 7; i++ )
		result = AlignStep
		(
			rawDepthMap, result,
			synthDepthBuffer, synthDepthBufferEyeToWorld,
			cameraParams
		) * result;

	return result;
}



util::float4x4 ICP::AlignStep
(
	util::vector2d< float > const & rawDepthMap,
	util::float4x4 const & rawDepthMapEyeToWorldGuess,
		
	util::vector2d< VertexPositionNormal > const & synthDepthBuffer,
	util::float4x4 const & synthDepthBufferEyeToWorld,

	DepthSensorParams const & cameraParams
)
{
	using namespace util;

	float4x4 dstWorldToEye = invert_transform( synthDepthBufferEyeToWorld );
	float4x4 srcWorldToEye = invert_transform( rawDepthMapEyeToWorldGuess );

	float4 flInv( 1.0f / cameraParams.FocalLengthPixels().x(), 1.0f / cameraParams.FocalLengthPixels().y(), 1.0f, 1.0f );
	float4 ppOverFl = float4
	(
		(0.5f - cameraParams.PrincipalPointPixels().x()),
		(cameraParams.PrincipalPointPixels().y() - 0.5f),
		0.0f, 
		0.0f
	) * flInv;

	kahan_sum< float4 > srcMedianSum( float4( 0.0f ) );
	kahan_sum< float4 > dstMedianSum( float4( 0.0f ) );
	m_assocs.clear();
	
	for( std::size_t y = 0; y < rawDepthMap.height(); y++ )
		for( std::size_t x = 0; x < rawDepthMap.width(); x++ )
		{
#if 1
			auto dst = synthDepthBuffer( x, y );
			if( 0.0f == dst.position.x() && 0.0f == dst.position.y() && 0.0f == dst.position.z() ) // invalid point
				continue;

			// project dst onto src
			float4 uv = homogenize( cameraParams.EyeToClipRH() * srcWorldToEye * float4( dst.position, 1.0f ) );

			// construct 3D point out of uv and raw depth (640x480)
			int u = (int) ( uv.x() * 320.0f + 320.0f );
			int v = 479 - (int) ( uv.y() * 240.0f + 240.0f );

			if( u < 0 || u > 639 ||
				v < 0 || v > 479 )
				continue;

			float depth = rawDepthMap( u, v );
			if( 0.0f == depth ) // invalid depth
				continue;

			float4 src( (float) u, - (float) v, -1.0f, 0.0f );

			src = ( src * flInv + ppOverFl ) * depth;
			src.w() = 1.0f;

			src = rawDepthMapEyeToWorldGuess * src;
			// TODO: Make this work with 0 assocs (return identity)
			// refine using compatibility and weighting, also stop on error threshold/count
			// consider stealing FuSci impl (not as expensive as I thought because A can be computed on the fly)

			// check compatibility
			float dist = length( float4( dst.position, 1.0f ) - src );
			if( dist > 0.1f ) // further than 10cm apart
				continue;

			// construct virtual point to reflect point-to-plane distance
			float4 pdst( dst.position, 1.0f );
			float4 n( dst.normal, 0.0f );
			n = ( n - 0.5f ) * 2.0f;
			n.w() = 0.0f;
			n /= length( n );
			float4 virt = src - dot( src - pdst, n ) * n;

			m_assocs.push_back( std::make_pair( src.xyz(), virt.xyz() ) );
			srcMedianSum += src;
			dstMedianSum += virt;
#else
			float depth = rawDepthMap( x, y );

			if( 0.0f == depth )
				continue;
			
			float4 point( (float) x, - (float) y, -1.0f, 0.0f );

			point = ( point * flInv + ppOverFl ) * depth;
			point.w = 1.0f;

			point = rawDepthMapEyeToWorldGuess * point; // raw depth map pixel in 3D world space

			float4 uv = homogenize( cameraParams.EyeToClipRH() * dstWorldToEye * point );
			/*int u = (int) (uv.x * halfWidth + halfWidth);
			int v = (int) synthDepthBuffer.height() - 1 - (int) (uv.y * halfHeight + halfHeight);

			if( u >= 0 && u < synthDepthBuffer.width() &&
				v >= 0 && v < synthDepthBuffer.height() )
			{
				float4 dst = float4( synthDepthBuffer( u, v ), 1.0f );
				if( dst.x == 0 && dst.y == 0 && dst.z == 0 ) // invalid point
					continue;

				if( len2( dst - point ) > 0.01f ) // points more than 10cm apart => not compatible
					continue;

				m_assocs.push_back( std::make_pair( point.xyz(), dst.xyz() ) );
				srcMedianSum += point;
				dstMedianSum += dst;
			}*/

			int u = (int) (uv.x * halfWidth + halfWidth) - 10;
			int v = (int) synthDepthBuffer.height() - 1 - (int) (uv.y * halfHeight + halfHeight) - 10;

			u = std::max( 0, u );
			v = std::max( 0, v );

			float mind = 0.01f;
			float4 p1, p2;
			for( int y1 = v; y1 < std::min( 480, v + 21 ); y1++ )
			for( int x1 = u; x1 < std::min( 640, u + 21 ); x1++ )
			{
				float4 dst = float4( synthDepthBuffer( x1, y1 ).position, 1.0f );
				if( dst.x == 0 && dst.y == 0 && dst.z == 0 ) // invalid point
					continue;

				if( len2( dst - point ) < mind )
				{
					mind = len2( dst - point );
					p1 = point;
					p2 = dst;
				}

				//m_assocs.push_back( std::make_pair( point.xyz(), dst.xyz() ) );
				//srcMedianSum += point;
				//dstMedianSum += dst;
			}
			if( mind < 0.01f )
			{
				m_assocs.push_back( std::make_pair( p1.xyz(), p2.xyz() ) );
				srcMedianSum += p1;
				dstMedianSum += p2;
			}
#endif
		}
	std::printf( "nassocs: %d\n", m_assocs.size() );
	// Simple test: Create small array of random points,
	// use it twice, just with different transforms (rigid),
	// so we know the exact assocs
	// apply identity to one set and a rigid to other (take depth stream worldToView)
	// assume identity for src and see if horn produces worldToView (after all that's what
	// you need to multiply src with to reach dst !!!

	/*kahan_sum< float4 > srcMedianSum( 0.0f );
	kahan_sum< float4 > dstMedianSum( 0.0f );
	m_assocs.clear();

	std::vector< float4 > testArray( 100 );
	for( int i = 0; i < testArray.size(); i++ )
		testArray[ i ] = float4
		(
			std::rand() / (float) RAND_MAX,
			std::rand() / (float) RAND_MAX,
			std::rand() / (float) RAND_MAX,
			1.0f
		);

	for( int i = 0; i < testArray.size(); i++ )
	{
		float4 src = testArray[ i ];
		float4 dst = rawDepthMapEyeToWorldGuess * src; // this should be the result of Horn!
		
		srcMedianSum += src;
		dstMedianSum += dst;

		m_assocs.push_back( std::make_pair( src.xyz(), dst.xyz() ) );
	}*/

	float4 srcMedian = (float4) srcMedianSum / (float) m_assocs.size();
	float4 dstMedian = (float4) dstMedianSum / (float) m_assocs.size();

	kahan_sum< float > 
		Sxx( 0.0f ), Sxy( 0.0f ), Sxz( 0.0f ),
		Syx( 0.0f ), Syy( 0.0f ), Syz( 0.0f ),
		Szx( 0.0f ), Szy( 0.0f ), Szz( 0.0f );

	// associations computed => Horn
	for( std::size_t i = 0; i < m_assocs.size(); ++i )
	{
		// all points are valid and compatible, no-brainer
		auto x = m_assocs[ i ];

		float4 src( x.first , 1.0f );
		float4 dst( x.second, 1.0f );

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

	// N is symmetric by construction.
	N(1, 0) = N(0, 1);
	N(2, 0) = N(0, 2);
	N(2, 1) = N(1, 2);
	N(3, 0) = N(0, 3);
	N(3, 1) = N(1, 3);
	N(3, 2) = N(2, 3);

	float4 ev;
	eigen( N, ev, N );

	int imaxev = 0;
    for (int ii = 1; ii < 4; ii++)
        if (ev[ii] > ev[imaxev])
            imaxev = ii;

	float4 q;
	switch( imaxev )
	{
		case 0: q = N.cols[ 0 ]; break;
		case 1: q = N.cols[ 1 ]; break;
		case 2: q = N.cols[ 2 ]; break;
		case 3: q = N.cols[ 3 ]; break;
	}
	q = normalize( q );

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

	float4 t = dstMedian - R * srcMedian;

	// translation matrix:
	float4x4 T = float4x4::identity();
	T.cols[ 3 ] = t;
	T.cols[ 3 ].w() = 1.0f;

	float4x4 result = T * R;
	return result;
}

} // namespace