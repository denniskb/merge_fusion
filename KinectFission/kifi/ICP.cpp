#include <kifi/util/numeric.h>

#include <kifi/ICP.h>

#pragma warning( push )
#pragma warning( disable : 4324 4458 4996 )
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#pragma warning( pop )

using namespace kifi;

#pragma warning( disable : 4100 4505 ) // unreferenced function due to toggling between myICP and pclICP



static util::float4x4 AlignStep
(
	util::vector2d< float > const & rawDepthMap,
	util::float4x4 const & rawDepthMapEyeToWorldGuess,
		
	util::vector2d< util::float3 > const & synthDepthBuffer,
	util::float4x4 const & synthDepthBufferEyeToWorld,

	DepthSensorParams const & cameraParams,

	std::vector< std::pair< util::float3, util::float3 > > & tmpAssocs
);

static void FindAssocs
(
	util::vector2d< float > const & rawDepthMap,
	util::float4x4 const & rawDepthMapEyeToWorldGuess,
		
	util::vector2d< util::float3 > const & synthDepthBuffer,
	util::float4x4 const & synthDepthBufferEyeToWorld,

	DepthSensorParams const & cameraParams,

	std::vector< std::pair< util::float3, util::float3 > > & outAssocs
);

/**
 * Given a list of associations between pairs of 3D points,
 * find the rigid transformation that minimizes the squared error between them,
 * using Horn's method (http://people.csail.mit.edu/bkph/papers/Absolute_Orientation.pdf).
 */
static util::float4x4 FindTransformHorn( std::vector< std::pair< util::float3, util::float3 > > const & assocs );

/**
 * Use PCL implementation of ICP to align both point clouds
 */
static util::float4x4 AlignPCL
(
	util::vector2d< float > const & rawDepthMap,
	util::float4x4 const & rawDepthMapEyeToWorldGuess,
		
	util::vector2d< util::float3 > const & synthDepthBuffer,

	DepthSensorParams const & cameraParams
);

class extrapolate
{
public:
	extrapolate( util::float2 fl, util::float2 pp, std::size_t width, std::size_t height ) :
		flInv( 1.0f / fl.x, 1.0f / fl.y, 1.0f, 1.0f ),
		ppOverFl( util::float4( 0.5f - pp.x, pp.y - 0.5f, 0.0f, 0.0f ) * flInv ),
		halfWidth( 0.5f * width ),
		halfHeight( 0.5f * height )
	{
	}

	util::float4 operator()( std::size_t x, std::size_t y, float d, util::float4x4 const & eyeToWorld )
	{
		util::float4 point( (float) x, - (float) y, -1.0f, 0.0f );

		point = ( point * flInv + ppOverFl ) * d;
		point.w = 1.0f;

		point = eyeToWorld * point;

		return point;
	}

private:
	util::float4 flInv;
	util::float4 ppOverFl;
	float halfWidth;
	float halfHeight;
};



namespace kifi {

util::float4x4 ICP::Align
(
	util::vector2d< float > const & rawDepthMap,
	util::float4x4 const & rawDepthMapEyeToWorldGuess,
		
	util::vector2d< util::float3 > const & synthDepthBuffer,
	util::float4x4 const & synthDepthBufferEyeToWorld,

	DepthSensorParams const & cameraParams
)
{
#if 1 // my ICP
	util::float4x4 result = rawDepthMapEyeToWorldGuess;
	
	for( int i = 0; i < 7; i++ )
		result = AlignStep
		(
			rawDepthMap, result,
			synthDepthBuffer, synthDepthBufferEyeToWorld,
			cameraParams,
			m_assocs
		) * result;

	return result;
#else // PCL ICP
	return AlignPCL( rawDepthMap, rawDepthMapEyeToWorldGuess, synthDepthBuffer, cameraParams );
#endif
}

} // namespace



util::float4x4 AlignStep
(
	util::vector2d< float > const & rawDepthMap,
	util::float4x4 const & rawDepthMapEyeToWorldGuess,
		
	util::vector2d< util::float3 > const & synthDepthBuffer,
	util::float4x4 const & synthDepthBufferEyeToWorld,

	DepthSensorParams const & cameraParams,

	std::vector< std::pair< util::float3, util::float3 > > & tmpAssocs
)
{
	using namespace util;

	tmpAssocs.clear();

	FindAssocs
	(
		rawDepthMap,
		rawDepthMapEyeToWorldGuess,
		synthDepthBuffer,
		synthDepthBufferEyeToWorld,
		cameraParams,
		tmpAssocs
	);

	std::printf( "#assocs: %llu\n", tmpAssocs.size() );

	return FindTransformHorn( tmpAssocs );
}

void FindAssocs
(
	util::vector2d< float > const & rawDepthMap,
	util::float4x4 const & rawDepthMapEyeToWorldGuess,
		
	util::vector2d< util::float3 > const & synthDepthBuffer,
	util::float4x4 const & synthDepthBufferEyeToWorld,

	DepthSensorParams const & cameraParams,

	std::vector< std::pair< util::float3, util::float3 > > & outAssocs
)
{
	using namespace util;

	float const dist_thres = 0.01f;

	float4x4 dstWorldToEye = synthDepthBufferEyeToWorld; invert_transform( dstWorldToEye );

	extrapolate eye2world
	(
		cameraParams.FocalLengthPixels(),
		cameraParams.PrincipalPointPixels(),
		synthDepthBuffer.width(),
		synthDepthBuffer.height()
	);

	float halfWidth  = synthDepthBuffer.width () * 0.5f;
	float halfHeight = synthDepthBuffer.height() * 0.5f;

	for( std::size_t y = 0; y < rawDepthMap.height(); y++ )
		for( std::size_t x = 0; x < rawDepthMap.width(); x += 4 )
		{
			float depth = rawDepthMap( x, y );

			if( 0.0f == depth )
				continue;

			float4 point = eye2world( x, y, depth, rawDepthMapEyeToWorldGuess );
			
			float4 uv = homogenize( cameraParams.EyeToClipRH() * dstWorldToEye * point );

			int u = (int) (uv.x * halfWidth + halfWidth) - 10;
			int v = (int) synthDepthBuffer.height() - 1 - (int) (uv.y * halfHeight + halfHeight) - 10;

			u = std::max( 0, u );
			v = std::max( 0, v );

			float mind = dist_thres;
			float4 p1, p2;
			for( int y1 = v; y1 < std::min( (int) rawDepthMap.height(), v + 21 ); y1++ )
			for( int x1 = u; x1 < std::min( (int) rawDepthMap.width() , u + 21 ); x1++ )
			{
				float4 dst = float4( synthDepthBuffer( x1, y1 ), 1.0f );
				if( dst.x == 0 && dst.y == 0 && dst.z == 0 ) // invalid point
					continue;

				if( len( dst - point ) < mind )
				{
					mind = len( dst - point );
					p1 = point;
					p2 = dst;
				}
			}

			if( mind < dist_thres )
			{
				outAssocs.push_back( std::make_pair( p1.xyz(), p2.xyz() ) );
			}
		}
}

static util::float4x4 FindTransformHorn( std::vector< std::pair< util::float3, util::float3 > > const & assocs )
{
	using namespace util;

	kahan_sum< float4 > srcMedianSum( 0.0f );
	kahan_sum< float4 > dstMedianSum( 0.0f );

	for( auto const & srcDst : assocs ) {
		srcMedianSum += float4( srcDst.first , 1.0f );
		dstMedianSum += float4( srcDst.second, 1.0f );
	}

	float4 srcMedian = srcMedianSum / float4( (float) assocs.size() );
	float4 dstMedian = dstMedianSum / float4( (float) assocs.size() );



	kahan_sum< float > 
		Sxx, Sxy, Sxz,
		Syx, Syy, Syz,
		Szx, Szy, Szz;

	for( auto const & srcDst : assocs ) {
		float4 src( srcDst.first , 1.0f );
		float4 dst( srcDst.second, 1.0f );

		src -= srcMedian;
		dst -= dstMedian;

		Sxx += src.x * dst.x;
        Sxy += src.x * dst.y;
        Sxz += src.x * dst.z;

        Syx += src.y * dst.x;
        Syy += src.y * dst.y;
        Syz += src.y * dst.z;

        Szx += src.z * dst.x;
        Szy += src.z * dst.y;
        Szz += src.z * dst.z;
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
		case 0: q = N.col0; break;
		case 1: q = N.col1; break;
		case 2: q = N.col2; break;
		case 3: q = N.col3; break;
	}
	q = normalize( q );

	float4x4 R = float4x4::identity();
    float xy = q.x * q.y;
    float xz = q.x * q.z;
    float xw = q.x * q.w;
    float yy = q.y * q.y;
    float yz = q.y * q.z;
    float yw = q.y * q.w;
    float zz = q.z * q.z;
    float zw = q.z * q.w;
    float ww = q.w * q.w;

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
	T.col3 = t;
	T.col3.w = 1.0f;

	return T * R;
}

static util::float4x4 AlignPCL
(
	util::vector2d< float > const & rawDepthMap,
	util::float4x4 const & rawDepthMapEyeToWorldGuess,
		
	util::vector2d< util::float3 > const & synthDepthBuffer,
	
	DepthSensorParams const & cameraParams
)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out (new pcl::PointCloud<pcl::PointXYZ>);

  // Fill in the CloudIn data
  cloud_in->width    = (int32_t) synthDepthBuffer.width();
  cloud_in->height   = (int32_t) synthDepthBuffer.height();
  cloud_in->is_dense = false;
  cloud_in->points.resize (cloud_in->width * cloud_in->height);

  for (size_t i = 0; i < cloud_in->points.size (); ++i)
  {
	auto p = synthDepthBuffer[ i ];
	cloud_in->points[i] = pcl::PointXYZ( p.x, p.y, p.z );
  }

  *cloud_out = *cloud_in;

  extrapolate depth2point( cameraParams.FocalLengthPixels(), cameraParams.PrincipalPointPixels(), rawDepthMap.width(), rawDepthMap.height() );

  for (size_t i = 0; i < cloud_out->points.size (); ++i)
  {
	  size_t x = i % rawDepthMap.width();
	  size_t y = i / rawDepthMap.width();

	auto p = depth2point( x, y, rawDepthMap[ i ], rawDepthMapEyeToWorldGuess );
	cloud_out->points[i] = pcl::PointXYZ( p.x, p.y, p.z );
  }

  Eigen::Matrix4f T;
  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
  icp.setInputSource(cloud_in);
  icp.setInputTarget(cloud_out);
  icp.setMaximumIterations(2);
  pcl::PointCloud<pcl::PointXYZ> Final;
  icp.align(Final, T);
  std::cout << "has converged:" << icp.hasConverged() << " score: " <<
  icp.getFitnessScore() << std::endl;
  std::cout << T << std::endl;

	// TODO: Correct conversion?
	return util::float4x4
	(
		T( 0, 0 ), T( 0, 1 ), T( 0, 2 ), T( 0, 3 ),
		T( 1, 0 ), T( 1, 1 ), T( 1, 2 ), T( 1, 3 ),
		T( 2, 0 ), T( 2, 1 ), T( 2, 2 ), T( 2, 3 ),
		T( 3, 0 ), T( 3, 1 ), T( 3, 2 ), T( 3, 3 )
	);
}