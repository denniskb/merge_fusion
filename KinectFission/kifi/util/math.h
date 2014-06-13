#pragma once

#include <cmath>

#include <tmmintrin.h>



namespace kifi {
namespace util {

#pragma region float4, float4x4

__declspec( align( 16 ) )
struct float4
{
	float x, y, z, w;

	inline float4();
	inline explicit float4( float s );
	inline float4( float x, float y, float z, float w );
	inline explicit float4( float const * src );

	inline operator float *();
	inline operator float const *() const;
};

struct float4x4
{
	float4 row0, row1, row2, row3;

	inline float4x4();
	inline float4x4
	(
		float m00, float m01, float m02, float m03,
		float m10, float m11, float m12, float m13,
		float m20, float m21, float m22, float m23,
		float m30, float m31, float m32, float m33
	);
	inline float4x4( float4 row0, float4 row1, float4 row2, float4 row3 );

	// src must be in row-major layout
	inline explicit float4x4( float const * src );
};

#pragma endregion

inline float4 cross           ( float4 u, float4 v );
inline float  dot             ( float4 u, float4 v );
inline float  dot3            ( float4 u, float4 v );
inline float4 homogenize      ( float4 v );
inline float  length          ( float4 v );
inline float  length3         ( float4 v );
inline float  length_squared  ( float4 v );
inline float  length_squared3 ( float4 v );
inline float4 normalize       ( float4 v );
inline float4 normalize3      ( float4 v );
	
inline float4x4 invert( float4x4 m );
inline float4x4 perspective_fov_rh( float fovYradians, float aspectWbyH, float nearZdistance, float farZdistance );



struct FPU
{
	typedef float4   vector;
	struct matrix { vector col0, col1, col2, col3; };

	static inline vector set  ( float s );
	static inline vector set  ( float x, float y, float z, float w );
	static inline vector load ( float4 src );
	static inline float4 store( vector src );
	
	static inline matrix   load ( float4x4 src );
	static inline float4x4 store( matrix   src );

	static inline vector add( vector u, vector v );
	static inline vector sub( vector u, vector v );
	static inline vector mul( vector u, vector v );
	static inline vector div( vector u, vector v );
	
	static inline vector mul( vector v, matrix m );
	static inline matrix mul( matrix m, matrix n );

	static inline vector cross         ( vector u, vector v );
	static inline vector dot           ( vector u, vector v );
	static inline vector homogenize    ( vector v );
	static inline vector length        ( vector v );
	static inline vector length_squared( vector v );
	static inline vector normalize     ( vector v );
};



struct SSE3
{
	typedef __m128 vector;	
	struct matrix { vector row0, row1, row2, row3; };

	static inline vector set  ( float s );
	static inline vector set  ( float x, float y, float z, float w );
	static inline vector load ( float4 src );
	static inline float4 store( vector src );
	
	static inline matrix   load ( float4x4 src );
	static inline float4x4 store( matrix   src );
	
	static inline vector add( vector u, vector v );
	static inline vector sub( vector u, vector v );
	static inline vector mul( vector u, vector v );
	static inline vector div( vector u, vector v );
	
	static inline vector mul( vector v, matrix m );
	static inline matrix mul( matrix m, matrix n );

	static inline vector cross         ( vector u, vector v );
	static inline vector dot           ( vector u, vector v );
	static inline vector homogenize    ( vector v );
	static inline vector length        ( vector v );
	static inline vector length_squared( vector v );
	static inline vector normalize     ( vector v );
};

}} // namespace



inline float4 operator+( float s, float4 v );
inline float4 operator-( float s, float4 v );
inline float4 operator*( float s, float4 v );
inline float4 operator/( float s, float4 v );

inline float4 operator+( float4 v, float s );
inline float4 operator-( float4 v, float s );
inline float4 operator*( float4 v, float s );
inline float4 operator/( float4 v, float s );

inline float4 & operator+=( float4 & v, float s );
inline float4 & operator-=( float4 & v, float s );
inline float4 & operator*=( float4 & v, float s );
inline float4 & operator/=( float4 & v, float s );

inline float4 operator+( float4 u, float4 v );
inline float4 operator-( float4 u, float4 v );
inline float4 operator*( float4 u, float4 v );
inline float4 operator/( float4 u, float4 v );

inline float4 & operator+=( float4 & u, float4 v );
inline float4 & operator-=( float4 & u, float4 v );
inline float4 & operator*=( float4 & u, float4 v );
inline float4 & operator/=( float4 & u, float4 v );

inline float4   operator*( float4   v, float4x4 m );
inline float4x4 operator*( float4x4 m, float4x4 n );

inline float4   & operator*=( float4   & v, float4x4 m );
inline float4x4 & operator*=( float4x4 & m, float4x4 n );

inline bool operator==( float4 u, float4 v );



#pragma region Implementation



namespace kifi {
namespace util {

#pragma region float4

float4::float4() 
{
}

float4::float4( float s ) : 
	x( s ), y( s ), z( s ), w( s )
{
}

float4::float4( float x, float y, float z, float w ) : 
	x( x ), y( y ), z( z ), w( w ) 
{
}

float4::float4( float const * src ) :
	x( src[ 0 ] ), y( src[ 1 ] ), z( src[ 2 ] ), w( src[ 3 ] )
{
}

float4::operator float *() 
{
	return & x;
}

float4::operator float const *() const 
{
	return & x;
}

#pragma endregion

#pragma region float4x4

float4x4::float4x4
(
	float m00, float m01, float m02, float m03,
	float m10, float m11, float m12, float m13,
	float m20, float m21, float m22, float m23,
	float m30, float m31, float m32, float m33
) :
	row0( m00, m01, m02, m03 ),
	row1( m10, m11, m12, m13 ),
	row2( m20, m21, m22, m23 ),
	row3( m30, m31, m32, m33 )
{
}

float4x4::float4x4( float4 row0, float4 row1, float4 row2, float4 row3 ) :
	row0( row0 ), 
	row1( row1 ), 
	row2( row2 ), 
	row3( row3 )
{
}

float4x4::float4x4( float const * src ) :
	row0( & src[  0 ] ),
	row1( & src[  4 ] ),
	row2( & src[  8 ] ),
	row3( & src[ 12 ] )
{
}

#pragma endregion

#pragma region free functions

float4 cross( float4 u, float4 v )
{
	return float4
	(
		u.y * v.z - u.z * v.y,
		u.z * v.x - u.x * v.z,
		u.x * v.y - u.y * v.x,
		0.0f
	);
}

float dot( float4 u, float4 v )
{
	return ( u.x * v.x + u.y * v.y ) + ( u.z * v.z + u.w * v.w );
}

float dot3( float4 u, float4 v )
{
	return u.x * v.x + u.y * v.y + u.z * v.z;
}

float4 homogenize( float4 v )
{
	return div( v, v.w );
}

float length( float4 v )
{
	return std::sqrt( length_squaredf( v ) );
}

float length3( float4 v )
{
	return std::sqrt( length_squared3f( v ) );
}

float length_squared( float4 v )
{
	return dotf( v, v );
}

float length_squared3( float4 v )
{
	return dot3f( v, v );
}

float4 normalize( float4 v )
{
	return div( v, lengthf( v ) );
}

float4 normalize3( float4 v )
{
	return div( v, length3f( v ) );
}

//float4x4 invert( float4x4 m ){...}
//float4x4 perspective_fov_rh( float fovYradians, float aspectWbyH, float nearZdistance, float farZdistance ){...}

#pragma endregion
	
#pragma region FPU backend

fpu::vector FPU::set( float s )
{
	return vector( s );
}

FPU::vector FPU::set ( float x, float y, float z, float w )
{
	return vector( x, y, z, w );
}

FPU::vector FPU::load( float4 src )
{
	return src;
}

float4 FPU::store( vector src )
{
	return src;
}

FPU::matrix FPU::load( float4x4 src )
{
	matrix result;

	result.col0.x = src.row0.x;
	result.col0.y = src.row1.x;
	result.col0.z = src.row2.x;
	result.col0.w = src.row3.x;

	result.col1.x = src.row0.y;
	result.col1.y = src.row1.y;
	result.col1.z = src.row2.y;
	result.col1.w = src.row3.y;

	result.col2.x = src.row0.z;
	result.col2.y = src.row1.z;
	result.col2.z = src.row2.z;
	result.col2.w = src.row3.z;

	result.col3.x = src.row0.w;
	result.col3.y = src.row1.w;
	result.col3.z = src.row2.w;
	result.col3.w = src.row3.w;

	return result;
}

float4x4 FPU::store( matrix src )
{
	float4x4 result;

	result.row0.x = src.col0.x;
	result.row0.y = src.col1.x;
	result.row0.z = src.col2.x;
	result.row0.w = src.col3.x;
		   
	result.row1.x = src.col0.y;
	result.row1.y = src.col1.y;
	result.row1.z = src.col2.y;
	result.row1.w = src.col3.y;
		   
	result.row2.x = src.col0.z;
	result.row2.y = src.col1.z;
	result.row2.z = src.col2.z;
	result.row2.w = src.col3.z;
		   
	result.row3.x = src.col0.w;
	result.row3.y = src.col1.w;
	result.row3.z = src.col2.w;
	result.row3.w = src.col3.w;

	return result;
}

FPU::vector FPU::add( vector u, vector v )
{
	return u + v;
}

FPU::vector FPU::sub( vector u, vector v )
{
	return u - v;
}

FPU::vector FPU::mul( vector u, vector v )
{
	return u * v;
}

FPU::vector FPU::div( vector u, vector v )
{
	return u / v;
}

FPU::vector FPU::mul( vector v, matrix m )
{
	return vector
	(
		util::dot( v, m.col0 ),
		util::dot( v, m.col1 ),
		util::dot( v, m.col2 ),
		util::dot( v, m.col3 )
	);
}

FPU::matrix FPU::mul( matrix m, matrix n )
{
	matrix result;
	
	result.col0 = ( n.col0.x * m.col0 + n.col0.y * m.col1 ) + ( n.col0.z * m.col2 + n.col0.w * m.col3 );
	result.col1 = ( n.col1.x * m.col0 + n.col1.y * m.col1 ) + ( n.col1.z * m.col2 + n.col1.w * m.col3 );
	result.col2 = ( n.col2.x * m.col0 + n.col2.y * m.col1 ) + ( n.col2.z * m.col2 + n.col2.w * m.col3 );
	result.col3 = ( n.col3.x * m.col0 + n.col3.y * m.col1 ) + ( n.col3.z * m.col2 + n.col3.w * m.col3 );
	
	return result;
}

FPU::vector FPU::cross( vector u, vector v )
{
	return util::cross( u, v );
}

FPU::vector FPU::dot( vector u, vector v )
{
	return util::dot( u, v );
}

FPU::vector FPU::homogenize( vector v )
{
	return util::homogenize( v );
}

FPU::vector FPU::length( vector v )
{
	return util::length( v );
}

FPU::vector FPU::length_squared( vector v )
{
	return util::length_squared( v );
}

FPU::vector FPU::normalize( vector v )
{
	return util::normalize( v );
}

#pragma endregion

#pragma region SSE3 backend

SSE3::vector SSE3::set( float s )
{
	return load( float4( s ) );
}

SSE3::vector SSE3::set( float x, float y, float z, float w )
{
	return load( float4( x, y, z, w ) );
}

SSE3::vector SSE3::load( float4 src )
{
	return _mm_load_ps( src );
}

float4 SSE3::store( vector src )
{
	float4 result;

	_mm_store_ps( result, src );

	return result;
}

SSE3::matrix SSE3::load ( float4x4 src )
{
	matrix result;

	result.row0 = load( src.row0 );
	result.row1 = load( src.row1 );
	result.row2 = load( src.row2 );
	result.row3 = load( src.row3 );

	return result;
}

float4x4 SSE3::store( matrix src )
{
	float4x4 result;

	result.row0 = store( src.row0 );
	result.row1 = store( src.row1 );
	result.row2 = store( src.row2 );
	result.row3 = store( src.row3 );

	return result;
}

SSE3::vector SSE3::add( vector u, vector v )
{
	return _mm_add_ps( u, v );
}

SSE3::vector SSE3::sub( vector u, vector v )
{
	return _mm_sub_ps( u, v );
}

SSE3::vector SSE3::mul( vector u, vector v )
{
	return _mm_mul_ps( u, v );
}

SSE3::vector SSE3::div( vector u, vector v )
{
	return _mm_div_ps( u, v );
}

SSE3::vector SSE3::mul( vector v, matrix m )
{
	vector tmp0 = _mm_shuffle_ps( v, v, _MM_SHUFFLE( 0, 0, 0, 0 ) );
	vector tmp1 = _mm_shuffle_ps( v, v, _MM_SHUFFLE( 1, 1, 1, 1 ) );
	vector tmp2 = _mm_shuffle_ps( v, v, _MM_SHUFFLE( 2, 2, 2, 2 ) );
	vector tmp3 = _mm_shuffle_ps( v, v, _MM_SHUFFLE( 3, 3, 3, 3 ) );

	tmp0 = mul( tmp0, m.row0 );
	tmp1 = mul( tmp1, m.row1 );
	tmp2 = mul( tmp2, m.row2 );
	tmp3 = mul( tmp3, m.row3 );

	tmp0 = add( tmp0, tmp1 );
	tmp2 = add( tmp2, tmp3 );

	return add( tmp0, tmp2 );
}

SSE3::matrix SSE3::mul( matrix m, matrix n )
{
	matrix result;

	result.row0 = mul( m.row0, n );
	result.row1 = mul( m.row1, n );
	result.row2 = mul( m.row2, n );
	result.row3 = mul( m.row3, n );

	return result;
}

SSE3::vector SSE3::cross( vector u, vector v )
{
	vector urot = _mm_shuffle_ps( u, u, _MM_SHUFFLE( 3, 0, 2, 1 ) );
	vector vrot = _mm_shuffle_ps( v, v, _MM_SHUFFLE( 3, 0, 2, 1 ) );

	urot = mul( u, vrot );
	vrot = mul( v, urot );

	urot = sub( urot, vrot );

	return _mm_shuffle_ps( urot, urot, _MM_SHUFFLE( 3, 0, 2, 1 ) );
}

SSE3::vector SSE3::dot( vector u, vector v )
{
	vector tmp = mul( u, v );

	tmp = _mm_hadd_ps( tmp, tmp );
	return _mm_hadd_ps( tmp, tmp );
}

SSE3::vector SSE3::homogenize( vector v )
{
	return div( v, _mm_shuffle_ps( v, v, _MM_SHUFFLE( 3, 3, 3, 3 ) ) );
}

SSE3::vector SSE3::length( vector v )
{
	return _mm_sqrt_ps( length_squared( v ) );
}

SSE3::vector SSE3::length_squared( vector v )
{
	return dot( v, v );
}

SSE3::vector SSE3::normalize( vector v )
{
	return div( v, length( v ) );
}

#pragma endregion

}} // namespace



#pragma region float4 and float4x4 operators

kifi::util::float4 operator+( float s, kifi::util::float4 v )
{
	return kifi::util::float4
	(
		s + v.x,
		s + v.y,
		s + v.z,
		s + v.w
	);
}

kifi::util::float4 operator-( float s, kifi::util::float4 v )
{
	return kifi::util::float4
	(
		s - v.x,
		s - v.y,
		s - v.z,
		s - v.w
	);
}

kifi::util::float4 operator*( float s, kifi::util::float4 v )
{
	return kifi::util::float4
	(
		s * v.x,
		s * v.y,
		s * v.z,
		s * v.w
	);
}

kifi::util::float4 operator/( float s, kifi::util::float4 v )
{
	return kifi::util::float4
	(
		s / v.x,
		s / v.y,
		s / v.z,
		s / v.w
	);
}

kifi::util::float4 operator+( kifi::util::float4 v, float s )
{
	return kifi::util::float4
	(
		v.x + s,
		v.y + s,
		v.z + s,
		v.w + s
	);
}

kifi::util::float4 operator-( kifi::util::float4 v, float s )
{
	return kifi::util::float4
	(
		v.x - s,
		v.y - s,
		v.z - s,
		v.w - s
	);
}

kifi::util::float4 operator*( kifi::util::float4 v, float s )
{
	return kifi::util::float4
	(
		v.x * s,
		v.y * s,
		v.z * s,
		v.w * s
	);
}

kifi::util::float4 operator/( kifi::util::float4 v, float s )
{
	float const sInv = 1.0f / s;

	return kifi::util::float4
	(
		v.x * sInv,
		v.y * sInv,
		v.z * sInv,
		v.w * sInv
	);
}

kifi::util::float4 & operator+=( kifi::util::float4 & v, float s )
{
	v = v + s;
	return v;
}

kifi::util::float4 & operator-=( kifi::util::float4 & v, float s )
{
	v = v - s;
	return v;
}

kifi::util::float4 & operator*=( kifi::util::float4 & v, float s )
{
	v = v * s;
	return v;
}

kifi::util::float4 & operator/=( kifi::util::float4 & v, float s )
{
	v = v / s;
	return v;
}

kifi::util::float4 operator+( kifi::util::float4 u, kifi::util::float4 v )
{
	return kifi::util::float4
	(
		u.x + v.x,
		u.y + v.y,
		u.z + v.z,
		u.w + v.w
	);
}

kifi::util::float4 operator-( kifi::util::float4 u, kifi::util::float4 v )
{
	return kifi::util::float4
	(
		u.x - v.x,
		u.y - v.y,
		u.z - v.z,
		u.w - v.w
	);
}

kifi::util::float4 operator*( kifi::util::float4 u, kifi::util::float4 v )
{
	return kifi::util::float4
	(
		u.x * v.x,
		u.y * v.y,
		u.z * v.z,
		u.w * v.w
	);
}

kifi::util::float4 operator/( kifi::util::float4 u, kifi::util::float4 v )
{
	return kifi::util::float4
	(
		u.x / v.x,
		u.y / v.y,
		u.z / v.z,
		u.w / v.w
	);
}

kifi::util::float4 & operator+=( kifi::util::float4 & u, kifi::util::float4 v )
{
	u = u + v;
	return u;
}

kifi::util::float4 & operator-=( kifi::util::float4 & u, kifi::util::float4 v )
{
	u = u - v;
	return u;
}

kifi::util::float4 & operator*=( kifi::util::float4 & u, kifi::util::float4 v )
{
	u = u * v;
	return u;
}

kifi::util::float4 & operator/=( kifi::util::float4 & u, kifi::util::float4 v )
{
	u = u / v;
	return u;
}

kifi::util::float4 operator*( kifi::util::float4 v, kifi::util::float4x4 m )
{
	return ( v.x * m.row0 + v.y * m.row1 ) + ( v.z * m.row2 + v.w * m.row3 );
}

kifi::util::float4x4 operator*( kifi::util::float4x4 m, kifi::util::float4x4 n )
{
	return kifi::util::float4x4
	(
		m.row0 * n,
		m.row1 * n,
		m.row2 * n,
		m.row3 * n
	);
}

kifi::util::float4 & operator*=( kifi::util::float4 & v, kifi::util::float4x4 m )
{
	v = v * m;
	return v;
}

kifi::util::float4x4 & operator*=( kifi::util::float4x4 & m, kifi::util::float4x4 n )
{
	m = m * n;
	return m;
}

bool operator==( kifi::util::float4 u, kifi::util::float4 v )
{
	return 
		u.x == v.x && 
		u.y == v.y && 
		u.z == v.z && 
		u.w == v.w;
}

#pragma endregion

#pragma endregion