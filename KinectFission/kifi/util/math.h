#pragma once

#ifdef KIFI_USE_FMA3
#include <immintrin.h> // AVX (only used for 'fma')
#endif

#include <xmmintrin.h> // SSE1



namespace kifi {
namespace util {

#pragma region float3/float4/float4x4/float4x4/vector/matrix

struct float3
{
	float x, y, z;

	inline float3();
	inline float3( float s );
	inline float3( float x, float y, float z );
};

struct float4
{
	float x, y, z, w;

	inline float4();
	inline float4( float s );
	inline float4( float x, float y, float z, float w );

	inline float3 xyz() const;
};



struct float4x4
{
	// stored in column-major order
	float 
		m00, m10, m20, m30,
		m01, m11, m21, m31,
		m02, m12, m22, m32,
		m03, m13, m23, m33;

	inline float4x4();
	inline float4x4
	(
		float m00, float m01, float m02, float m03,
		float m10, float m11, float m12, float m13,
		float m20, float m21, float m22, float m23,
		float m30, float m31, float m32, float m33
	);

	// src must be in row-major order
	inline explicit float4x4( float const * src );
};

float4x4 const identity
(
	1.0f, 0.0f, 0.0f, 0.0f,
	0.0f, 1.0f, 0.0f, 0.0f,
	0.0f, 0.0f, 1.0f, 0.0f,
	0.0f, 0.0f, 0.0f, 1.0f
);



typedef __m128 vector;
struct matrix { vector row0, row1, row2, row3; };

#pragma endregion

#pragma region Operators

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

inline vector   operator* ( vector v  , matrix m );
inline vector & operator*=( vector & v, matrix m );

#pragma endregion

#pragma region Functions

template< typename T >
T               clamp     ( T x, T a, T b );
inline float    dot       ( float4 u, float4 v );
inline float4   homogenize( float4 v );
template< typename T >
T               lerp      ( T a, T b, T weightB );
inline unsigned	pack      ( unsigned x, unsigned y, unsigned z );
inline bool     powerOf2  ( int x );
inline void     unpack    ( unsigned v, unsigned & outX, unsigned & outY, unsigned & outZ );

inline float4x4 invert_transform  ( float4x4 const & Rt );
inline float4x4 perspective_fov_rh( float fovYradians, float aspectWbyH, float nearZdistance, float farZdistance );
inline float4x4 transpose         ( float4x4 const & m );

inline int      all       ( vector v );
inline int      any       ( vector v );
template< int index >	  
inline vector   broadcast ( vector v );
inline vector   dot       ( vector u, vector v );
inline vector   fma       ( vector u, vector v, vector w );
inline vector   homogenize( vector v );
inline vector   load      ( float const * src );
inline vector   loadu     ( float const * src );
inline vector   loadss    ( float src );
inline int      none      ( vector v );
inline vector   set       ( float s );
inline vector   set       ( float x, float y, float z, float w );
inline vector   set       ( float4 v );
inline matrix   set       ( float4x4 m );
template< int a0, int a1, int b0, int b1 >
inline vector   shuffle   ( vector a, vector b );
inline void     store     ( float * dst, vector src );
inline float    storess   ( vector src );
inline void     storeu    ( float * dst, vector src );
inline vector   zero      ();

#pragma endregion

}} // namespace

#pragma region Global Operators

inline __m128 operator+( __m128 u, __m128 v );
inline __m128 operator-( __m128 u, __m128 v );
inline __m128 operator*( __m128 u, __m128 v );
inline __m128 operator/( __m128 u, __m128 v );
	   
inline __m128 & operator+=( __m128 & u, __m128 v );
inline __m128 & operator-=( __m128 & u, __m128 v );
inline __m128 & operator*=( __m128 & u, __m128 v );
inline __m128 & operator/=( __m128 & u, __m128 v );

inline __m128 operator==( __m128 u, __m128 v );
inline __m128 operator!=( __m128 u, __m128 v );
inline __m128 operator< ( __m128 u, __m128 v );
inline __m128 operator<=( __m128 u, __m128 v );
inline __m128 operator> ( __m128 u, __m128 v );
inline __m128 operator>=( __m128 u, __m128 v );

inline __m128 operator|( __m128 u, __m128 v );
inline __m128 operator&( __m128 u, __m128 v );

#pragma endregion



#pragma region Implementation

#include <cassert>
#include <cmath>
#include <cstring>



namespace kifi {
namespace util {

#pragma region float4/float4x4

float3::float3() {}
float3::float3( float s ) : x( s ), y( s ), z( s ) {}
float3::float3( float x, float y, float z ) : x( x ), y( y ), z( z ) {}

float4::float4() {}
float4::float4( float s ) : x( s ), y( s ), z( s ), w( s ) {}
float4::float4( float x, float y, float z, float w ) : x( x ), y( y ), z( z ), w( w ) {}
float3 float4::xyz() const { return float3( x, y, z ); }



float4x4::float4x4() {}

float4x4::float4x4
(
	float m00, float m01, float m02, float m03,
	float m10, float m11, float m12, float m13,
	float m20, float m21, float m22, float m23,
	float m30, float m31, float m32, float m33
) :
	m00( m00 ), m10( m10 ), m20( m20 ), m30( m30 ),
	m01( m01 ), m11( m11 ), m21( m21 ), m31( m31 ),
	m02( m02 ), m12( m12 ), m22( m22 ), m32( m32 ),
	m03( m03 ), m13( m13 ), m23( m23 ), m33( m33 )
{}

float4x4::float4x4( float const * src )
{
	std::memcpy( & m00, src, 64 );
	* this = transpose( * this );
}

#pragma endregion

#pragma region Operators

float4 operator+( float4 u, float4 v )
{
	return float4
	(
		u.x + v.x,
		u.y + v.y,
		u.z + v.z,
		u.w + v.w
	);
}

float4 operator-( float4 u, float4 v )
{
	return float4
	(
		u.x - v.x,
		u.y - v.y,
		u.z - v.z,
		u.w - v.w
	);
}

float4 operator*( float4 u, float4 v )
{
	return float4
	(
		u.x * v.x,
		u.y * v.y,
		u.z * v.z,
		u.w * v.w
	);
}

float4 operator/( float4 u, float4 v )
{
	return float4
	(
		u.x / v.x,
		u.y / v.y,
		u.z / v.z,
		u.w / v.w
	);
}



float4 & operator+=( float4 & u, float4 v )
{
	u = u + v;
	return u;
}

float4 & operator-=( float4 & u, float4 v )
{
	u = u - v;
	return u;
}

float4 & operator*=( float4 & u, float4 v )
{
	u = u * v;
	return u;
}

float4 & operator/=( float4 & u, float4 v )
{
	u = u / v;
	return u;
}



float4 operator*( float4 v, float4x4 m )
{
	return float4
	(
		(v.x * m.m00 + v.y * m.m10) + (v.z * m.m20 + v.w * m.m30),
		(v.x * m.m01 + v.y * m.m11) + (v.z * m.m21 + v.w * m.m31),
		(v.x * m.m02 + v.y * m.m12) + (v.z * m.m22 + v.w * m.m32),
		(v.x * m.m03 + v.y * m.m13) + (v.z * m.m23 + v.w * m.m33)
	);
}

float4x4 operator*( float4x4 m, float4x4 n )
{
	float4x4 result;

	result.m00 =  n.m00 * m.m00;
	result.m10 =  n.m00 * m.m10;
	result.m20 =  n.m00 * m.m20;
	result.m30 =  n.m00 * m.m30;

	result.m00 += n.m10 * m.m01;
	result.m10 += n.m10 * m.m11;
	result.m20 += n.m10 * m.m21;
	result.m30 += n.m10 * m.m31;

	result.m00 += n.m20 * m.m02;
	result.m10 += n.m20 * m.m12;
	result.m20 += n.m20 * m.m22;
	result.m30 += n.m20 * m.m32;

	result.m00 += n.m30 * m.m03;
	result.m10 += n.m30 * m.m13;
	result.m20 += n.m30 * m.m23;
	result.m30 += n.m30 * m.m33;



	result.m01 =  n.m01 * m.m00;
	result.m11 =  n.m01 * m.m10;
	result.m21 =  n.m01 * m.m20;
	result.m31 =  n.m01 * m.m30;

	result.m01 += n.m11 * m.m01;
	result.m11 += n.m11 * m.m11;
	result.m21 += n.m11 * m.m21;
	result.m31 += n.m11 * m.m31;

	result.m01 += n.m21 * m.m02;
	result.m11 += n.m21 * m.m12;
	result.m21 += n.m21 * m.m22;
	result.m31 += n.m21 * m.m32;

	result.m01 += n.m31 * m.m03;
	result.m11 += n.m31 * m.m13;
	result.m21 += n.m31 * m.m23;
	result.m31 += n.m31 * m.m33;



	result.m02 =  n.m02 * m.m00;
	result.m12 =  n.m02 * m.m10;
	result.m22 =  n.m02 * m.m20;
	result.m32 =  n.m02 * m.m30;

	result.m02 += n.m12 * m.m01;
	result.m12 += n.m12 * m.m11;
	result.m22 += n.m12 * m.m21;
	result.m32 += n.m12 * m.m31;

	result.m02 += n.m22 * m.m02;
	result.m12 += n.m22 * m.m12;
	result.m22 += n.m22 * m.m22;
	result.m32 += n.m22 * m.m32;

	result.m02 += n.m32 * m.m03;
	result.m12 += n.m32 * m.m13;
	result.m22 += n.m32 * m.m23;
	result.m32 += n.m32 * m.m33;



	result.m03 =  n.m03 * m.m00;
	result.m13 =  n.m03 * m.m10;
	result.m23 =  n.m03 * m.m20;
	result.m33 =  n.m03 * m.m30;

	result.m03 += n.m13 * m.m01;
	result.m13 += n.m13 * m.m11;
	result.m23 += n.m13 * m.m21;
	result.m33 += n.m13 * m.m31;

	result.m03 += n.m23 * m.m02;
	result.m13 += n.m23 * m.m12;
	result.m23 += n.m23 * m.m22;
	result.m33 += n.m23 * m.m32;

	result.m03 += n.m33 * m.m03;
	result.m13 += n.m33 * m.m13;
	result.m23 += n.m33 * m.m23;
	result.m33 += n.m33 * m.m33;

	return result;
}



vector operator*( vector v, matrix m )
{
	vector vx = broadcast< 0 >( v );
	vector vy = broadcast< 1 >( v );
	vector vz = broadcast< 2 >( v );
	vector vw = broadcast< 3 >( v );

	vx *= m.row0;
	vy *= m.row1;
	vz *= m.row2;
	vw *= m.row3;

	vx += vy;
	vz += vw;

	return vx + vz;
}

vector & operator*=( vector & v, matrix m )
{
	v = v * m;
	return v;
}

#pragma endregion

#pragma region Functions

template< typename T >
T clamp( T x, T a, T b )
{
	return std::max( a, std::min( x, b ) );
}

float dot( float4 u, float4 v )
{
	return (u.x * v.x + u.y * v.y) + (u.z * v.z + u.w * v.w);
}

float4 homogenize( float4 v )
{
	assert( v.w != 0.0f );

	return v / v.w;
}

template< typename T >
T lerp( T a, T b, T weightB )
{
	return a + (b-a) * weightB;
}

unsigned pack( unsigned x, unsigned y, unsigned z )
{
	return z << 20 | y << 10 | x;
}

bool powerOf2( int x )
{
	return x > 0 && ! (x & (x - 1));
}

void unpack( unsigned v, unsigned & outX, unsigned & outY, unsigned & outZ )
{
	outX = v & 0x3ff;
	outY = v >> 10 & 0x3ff;
	outZ = v >> 20;
}



float4x4 invert_transform( float4x4 const & Rt )
{
	float4x4 R( Rt );
	R.m30 = 0.0f;
	R.m31 = 0.0f;
	R.m32 = 0.0f;

	float4x4 tInv = identity;
	tInv.m30 = -Rt.m30;
	tInv.m31 = -Rt.m31;
	tInv.m32 = -Rt.m32;

	return tInv * transpose( R );
}

float4x4 perspective_fov_rh( float fovYradians, float aspectWbyH, float nearZdistance, float farZdistance )
{
	float h = 1.0f / std::tanf( 0.5f * fovYradians );
	float w = h / aspectWbyH;
	float Q = farZdistance / (farZdistance - nearZdistance);
	
	float4x4 result;
	std::memset( & result.m00, 0, 64 );

	result.m00 = w;
	result.m11 = h;
	result.m22 = -Q;
	result.m32 = -Q * nearZdistance;
	result.m23 = -1.0f;

	return result;
}

float4x4 transpose( float4x4 const & m )
{
	return float4x4
	(
		m.m00, m.m10, m.m20, m.m30,
		m.m01, m.m11, m.m21, m.m31,
		m.m02, m.m12, m.m22, m.m32,
		m.m03, m.m13, m.m23, m.m33
	);
}



int all( vector v )
{
	return ( 0xf == _mm_movemask_ps( v ) );
}

int any( vector v )
{
	return _mm_movemask_ps( v );
}

template< int i >
vector broadcast( vector v )
{
	return shuffle< i, i, i, i >( v, v );
}

vector dot( vector u, vector v )
{
	vector tmp = u * v;

	vector rotl1 = shuffle< 1, 2, 3, 0 >( tmp, tmp );
	vector rotl2 = shuffle< 2, 3, 0, 1 >( tmp, tmp );
	vector rotl3 = shuffle< 3, 0, 1, 2 >( tmp, tmp );

	tmp   += rotl1;
	rotl2 += rotl3;

	return tmp + rotl2;
}

vector fma( vector u, vector v, vector w )
{
#ifdef KIFI_USE_FMA3
	return _mm_fmadd_ps( u, v, w );
#else
	return u * v + w;
#endif
}

vector homogenize( vector v )
{
	return v / broadcast< 3 >( v );
}

vector load( float const * src )
{
	return _mm_load_ps( src );
}

vector loadu( float const * src )
{
	return _mm_loadu_ps( src );
}

vector loadss( float src )
{
	return _mm_load_ss( & src );
}

int none( vector v )
{
	return ( 0 == _mm_movemask_ps( v ) );
}


vector set( float s )
{
	return _mm_set1_ps( s );
}

vector set( float x, float y, float z, float w )
{
	return _mm_set_ps( w, z, y, x );
}

vector set( float4 v )
{
	return loadu( reinterpret_cast< float * >( & v ) );
}

matrix set( float4x4 m )
{
	matrix result;

	result.row0 = set( m.m00, m.m01, m.m02, m.m03 );
	result.row1 = set( m.m10, m.m11, m.m12, m.m13 );
	result.row2 = set( m.m20, m.m21, m.m22, m.m23 );
	result.row3 = set( m.m30, m.m31, m.m32, m.m33 );

	return result;
}

template< int a0, int a1, int b0, int b1 >
vector shuffle( vector a, vector b )
{
	return _mm_shuffle_ps( a, b, _MM_SHUFFLE( b1, b0, a1, a0 ) );
}

void store( float * dst, vector src )
{
	_mm_store_ps( dst, src );
}

float storess( vector src )
{
	float result;
	_mm_store_ss( & result, src );
	return result;
}

void storeu( float * dst, vector src )
{
	_mm_storeu_ps( dst, src );
}

vector zero()
{
	return _mm_setzero_ps();
}

#pragma endregion

}} // namespace

#pragma region Global Operators

__m128 operator+( __m128 u, __m128 v )
{
	return _mm_add_ps( u, v );
}

__m128 operator-( __m128 u, __m128 v )
{
	return _mm_sub_ps( u, v );
}

__m128 operator*( __m128 u, __m128 v )
{
	return _mm_mul_ps( u, v );
}

__m128 operator/( __m128 u, __m128 v )
{
	return _mm_div_ps( u, v );
}
	   


__m128 & operator+=( __m128 & u, __m128 v )
{
	u = u + v;
	return u;
}

__m128 & operator-=( __m128 & u, __m128 v )
{
	u = u - v;
	return u;
}

__m128 & operator*=( __m128 & u, __m128 v )
{
	u = u * v;
	return u;
}

__m128 & operator/=( __m128 & u, __m128 v )
{
	u = u / v;
	return u;
}



__m128 operator==( __m128 u, __m128 v )
{
	return _mm_cmpeq_ps( u, v );
}

__m128 operator!=( __m128 u, __m128 v )
{
	return _mm_cmpneq_ps( u, v );
}

__m128 operator<( __m128 u, __m128 v )
{
	return _mm_cmplt_ps( u, v );
}

__m128 operator<=( __m128 u, __m128 v )
{
	return _mm_cmple_ps( u, v );
}

__m128 operator>( __m128 u, __m128 v )
{
	return _mm_cmpgt_ps( u, v );
}

__m128 operator>=( __m128 u, __m128 v )
{
	return _mm_cmpge_ps( u, v );
}



__m128 operator|( __m128 u, __m128 v )
{
	return _mm_or_ps( u, v );
}

__m128 operator&( __m128 u, __m128 v )
{
	return _mm_and_ps( u, v );
}

#pragma endregion

#pragma endregion