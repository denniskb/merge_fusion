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



#pragma warning( push )
#pragma warning( disable : 4201 ) // non-standard extension used (unnamed struct/union)

struct float4x4
{
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

	inline float & operator()( int iRow, int iCol );
	inline float   operator()( int iRow, int iCol ) const;

	inline float4 row( int i ) const;
	inline float4 col( int i ) const;

private:
	float m_data[ 16 ]; // row-major
};

#pragma warning( pop )

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

inline vector operator* ( vector v, matrix m );

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
inline void     transpose         ( float4x4 & m );

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

#include <algorithm>
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
)
{
	(*this)( 0, 0 ) = m00;
	(*this)( 0, 1 ) = m01;
	(*this)( 0, 2 ) = m02;
	(*this)( 0, 3 ) = m03;

	(*this)( 1, 0 ) = m10;
	(*this)( 1, 1 ) = m11;
	(*this)( 1, 2 ) = m12;
	(*this)( 1, 3 ) = m13;

	(*this)( 2, 0 ) = m20;
	(*this)( 2, 1 ) = m21;
	(*this)( 2, 2 ) = m22;
	(*this)( 2, 3 ) = m23;

	(*this)( 3, 0 ) = m30;
	(*this)( 3, 1 ) = m31;
	(*this)( 3, 2 ) = m32;
	(*this)( 3, 3 ) = m33;
}

float4x4::float4x4( float const * src )
{
	std::memcpy( m_data, src, 64 );
}

float & float4x4::operator()( int iRow, int iCol )
{
	return m_data[ iCol + 4 * iRow ];
}

float float4x4::operator()( int iRow, int iCol ) const
{
	return m_data[ iCol + 4 * iRow ];
}

float4 float4x4::row( int i ) const
{
	return float4
	(
		(*this)( i, 0 ),
		(*this)( i, 1 ),
		(*this)( i, 2 ),
		(*this)( i, 3 )
	);
}

float4 float4x4::col( int i ) const
{
	return float4
	(
		(*this)( 0, i ),
		(*this)( 1, i ),
		(*this)( 2, i ),
		(*this)( 3, i )
	);
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
		dot( v, m.col( 0 ) ),
		dot( v, m.col( 1 ) ),
		dot( v, m.col( 2 ) ),
		dot( v, m.col( 3 ) )
	);
}

float4x4 operator*( float4x4 m, float4x4 n )
{
	float4x4 result;

	for( int row = 0; row < 4; row++ )
		for( int col = 0; col < 4; col++ )
			result( row, col ) = dot( m.row( row ), n.col( col ) );

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
	R( 3, 0 ) = 0.0f;
	R( 3, 1 ) = 0.0f;
	R( 3, 2 ) = 0.0f;

	float4x4 tInv = identity;
	tInv( 3, 0 ) = -Rt( 3, 0 );
	tInv( 3, 1 ) = -Rt( 3, 1 );
	tInv( 3, 2 ) = -Rt( 3, 2 );

	transpose( R );
	return tInv * R;
}

float4x4 perspective_fov_rh( float fovYradians, float aspectWbyH, float nearZdistance, float farZdistance )
{
	float h = 1.0f / std::tanf( 0.5f * fovYradians );
	float w = h / aspectWbyH;
	float Q = farZdistance / (farZdistance - nearZdistance);
	
	float4x4 result;
	std::memset( & result, 0, 64 );

	result( 0, 0 ) = w;
	result( 1, 1 ) = h;
	result( 2, 2 ) = -Q;
	result( 3, 2 ) = -Q * nearZdistance;
	result( 2, 3 ) = -1.0f;

	return result;
}

void transpose( float4x4 & m )
{
	for( int row = 0; row < 4; row++ )
		for( int col = row; col < 4; col++ )
			std::swap( m( row, col ), m( col, row ) );
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

	result.row0 = set( m.row( 0 ) );
	result.row1 = set( m.row( 1 ) );
	result.row2 = set( m.row( 2 ) );
	result.row3 = set( m.row( 3 ) );

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