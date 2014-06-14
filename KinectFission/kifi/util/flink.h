#pragma once

#include <cmath>

#include <immintrin.h>
#include <tmmintrin.h>



#pragma region Global Operators

inline __m128 operator+( __m128 u, __m128 v );
inline __m128 operator-( __m128 u, __m128 v );
inline __m128 operator*( __m128 u, __m128 v );
inline __m128 operator/( __m128 u, __m128 v );
	   
inline __m128 & operator+=( __m128 & u, __m128 v );
inline __m128 & operator-=( __m128 & u, __m128 v );
inline __m128 & operator*=( __m128 & u, __m128 v );
inline __m128 & operator/=( __m128 & u, __m128 v );

#pragma endregion

namespace flink {

struct float4;
struct float4x4;

#pragma region Operators

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

inline float4 operator- ( float4 v );

#pragma endregion

#pragma region Functions

inline float4 cross     ( float4 u, float4 v );
inline float  dot       ( float4 u, float4 v );
inline float4 homogenize( float4 v );
inline float  hsum      ( float4 v );
inline float  len       ( float4 v );
inline float  len_sq    ( float4 v );
inline float4 normalize ( float4 v );
	
inline float4x4 invert_transform( float4x4 Rt );
inline float4x4 transpose       ( float4x4 m );

inline float4x4 perspective_fov_rh( float fovYradians, float aspectWbyH, float nearZdistance, float farZdistance );

template< typename T >
inline T        clamp   ( T x, T a, T b );
template< typename T >
inline T        lerp    ( T a, T b, T weightB );
inline unsigned pack    ( unsigned x, unsigned y, unsigned z );
inline bool     powerOf2( unsigned x );
inline void     unpack  ( unsigned v, unsigned & outX, unsigned & outY, unsigned & outZ );

#pragma endregion

#pragma region FPU Backend

struct FPU
{
	typedef float4   vector;
	typedef float4x4 matrix;

	static inline vector set  ( float s );
	static inline vector set  ( float x, float y, float z, float w );
	static inline vector load ( float4 src );
	static inline float4 store( vector src );
	
	static inline matrix   load ( float4x4 src );
	static inline float4x4 store( matrix   src );

	static inline vector cross     ( vector u, vector v );
	static inline vector dot       ( vector u, vector v );
	static inline vector fma3      ( vector u, vector v, vector w );
	static inline vector homogenize( vector v );
	static inline vector hsum      ( vector v );
	static inline vector len       ( vector v );
	static inline vector len_sq    ( vector v );
	static inline vector normalize ( vector v );
};

// FPU::vector operator*( FPU::vector, FPU::matrix )
// FPU::matrix operator*( FPU::matrix, FPU::matrix )

// FPU::vector & operator*=( FPU::vector &, FPU::matrix )
// FPU::matrix & operator*=( FPU::matrix &, FPU::matrix )

#pragma endregion

#pragma region SSE3 Backend

struct SSE_FMA;
template< class FMA = SSE_FMA >
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

	static inline vector cross     ( vector u, vector v );
	static inline vector dot       ( vector u, vector v );
	static inline vector fma3      ( vector u, vector v, vector w );
	static inline vector homogenize( vector v );
	static inline vector hsum      ( vector v );
	static inline vector len       ( vector v );
	static inline vector len_sq    ( vector v );
	static inline vector normalize ( vector v );
};

template< class FMA >
inline typename SSE3< FMA >::vector operator*( typename SSE3< FMA >::vector v, typename SSE3< FMA >::matrix m );
template< class FMA >
inline typename SSE3< FMA >::matrix operator*( typename SSE3< FMA >::matrix m, typename SSE3< FMA >::matrix n );

template< class FMA >
inline typename SSE3< FMA >::vector & operator*=( typename SSE3< FMA >::vector & v, typename SSE3< FMA >::matrix m );
template< class FMA >
inline typename SSE3< FMA >::matrix & operator*=( typename SSE3< FMA >::matrix & m, typename SSE3< FMA >::matrix n );

struct SSE_FMA
{
	static inline __m128 fma3( __m128 a, __m128 b, __m128 c );
};

struct AVX_FMA
{
	static inline __m128 fma3( __m128 a, __m128 b, __m128 c );
};

#pragma endregion

#pragma region float4/float4x4

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



float4x4 const identity
(
	1.0f, 0.0f, 0.0f, 0.0f,
	0.0f, 1.0f, 0.0f, 0.0f,
	0.0f, 0.0f, 1.0f, 0.0f,
	0.0f, 0.0f, 0.0f, 1.0f
);

#pragma endregion

} // namespace



#pragma region Implementation

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

#pragma endregion

namespace flink {

#pragma region Operators

float4 operator+( float s, float4 v )
{
	return float4
	(
		s + v.x,
		s + v.y,
		s + v.z,
		s + v.w
	);
}

float4 operator-( float s, float4 v )
{
	return float4
	(
		s - v.x,
		s - v.y,
		s - v.z,
		s - v.w
	);
}

float4 operator*( float s, float4 v )
{
	return float4
	(
		s * v.x,
		s * v.y,
		s * v.z,
		s * v.w
	);
}

float4 operator/( float s, float4 v )
{
	return float4
	(
		s / v.x,
		s / v.y,
		s / v.z,
		s / v.w
	);
}



float4 operator+( float4 v, float s )
{
	return float4
	(
		v.x + s,
		v.y + s,
		v.z + s,
		v.w + s
	);
}

float4 operator-( float4 v, float s )
{
	return float4
	(
		v.x - s,
		v.y - s,
		v.z - s,
		v.w - s
	);
}

float4 operator*( float4 v, float s )
{
	return float4
	(
		v.x * s,
		v.y * s,
		v.z * s,
		v.w * s
	);
}

float4 operator/( float4 v, float s )
{
	float const sInv = 1.0f / s;

	return float4
	(
		v.x * sInv,
		v.y * sInv,
		v.z * sInv,
		v.w * sInv
	);
}



float4 & operator+=( float4 & v, float s )
{
	v.x += s;
	v.y += s;
	v.z += s;
	v.w += s;

	return v;
}

float4 & operator-=( float4 & v, float s )
{
	v.x -= s;
	v.y -= s;
	v.z -= s;
	v.w -= s;

	return v;
}

float4 & operator*=( float4 & v, float s )
{
	v.x *= s;
	v.y *= s;
	v.z *= s;
	v.w *= s;

	return v;
}

float4 & operator/=( float4 & v, float s )
{
	float const sInv = 1.0f / s;

	v.x *= sInv;
	v.y *= sInv;
	v.z *= sInv;
	v.w *= sInv;

	return v;
}



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
	u.x += v.x;
	u.y += v.y;
	u.z += v.z;
	u.w += v.w;

	return u;
}

float4 & operator-=( float4 & u, float4 v )
{
	u.x -= v.x;
	u.y -= v.y;
	u.z -= v.z;
	u.w -= v.w;

	return u;
}

float4 & operator*=( float4 & u, float4 v )
{
	u.x *= v.x;
	u.y *= v.y;
	u.z *= v.z;
	u.w *= v.w;

	return u;
}

float4 & operator/=( float4 & u, float4 v )
{
	u.x /= v.x;
	u.y /= v.y;
	u.z /= v.z;
	u.w /= v.w;

	return u;
}



float4 operator*( float4 v, float4x4 m )
{
	return ( v.x * m.row0 + v.y * m.row1 ) + ( v.z * m.row2 + v.w * m.row3 );
}

float4x4 operator*( float4x4 m, float4x4 n )
{
	return float4x4
	(
		m.row0 * n,
		m.row1 * n,
		m.row2 * n,
		m.row3 * n
	);
}

float4 & operator*=( float4 & v, float4x4 m )
{
	v = v * m;
	return v;
}

float4x4 & operator*=( float4x4 & m, float4x4 n )
{
	m.row0 *= n;
	m.row1 *= n;
	m.row2 *= n;
	m.row3 *= n;

	return m;
}



float4 operator-( float4 v )
{
	return float4( -v.x, -v.y, -v.z, -v.w );
}

#pragma endregion

#pragma region Functions

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
	return hsum( u * v );
}

float4 fma3( float4 u, float4 v, float4 w )
{
	return u * v + w;
}

float4 homogenize( float4 v )
{
	return v / v.w;
}

float hsum( float4 v )
{
	return ( v.x + v.y ) + ( v.z + v.w );
}

float len( float4 v )
{
	return std::sqrt( len_sq( v ) );
}

float len_sq( float4 v )
{
	return dot( v, v );
}

float4 normalize( float4 v )
{
	return v / len( v );
}



float4x4 invert_transform( float4x4 Rt )
{
	float4x4 R = Rt;
	R.row3 = float4( 0.0f, 0.0f, 0.0f, 1.0f );

	float4x4 tInv = identity;
	tInv.row3 = -Rt.row3;

	return tInv * transpose( R );
}

float4x4 transpose( float4x4 m )
{
	return float4x4
	(
		m.row0.x, m.row1.x, m.row2.x, m.row3.x,
		m.row0.y, m.row1.y, m.row2.y, m.row3.y,
		m.row0.z, m.row1.z, m.row2.z, m.row3.z,
		m.row0.w, m.row1.w, m.row2.w, m.row3.w
	);
}



//float4x4 perspective_fov_rh( float fovYradians, float aspectWbyH, float nearZdistance, float farZdistance ){...}



template< typename T >
T clamp( T x, T a, T b )
{
	return std::max( a, std::min( x, b ) );
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

bool powerOf2( unsigned x )
{
	return x > 0 && ! (x & (x - 1));
}

void unpack( unsigned v, unsigned & outX, unsigned & outY, unsigned & outZ )
{
	outX = v & 0x3ff;
	outY = v >> 10 & 0x3ff;
	outZ = v >> 20;
}

#pragma endregion

#pragma region FPU Backend

FPU::vector FPU::set( float s )
{
	return vector( s );
}

FPU::vector FPU::set( float x, float y, float z, float w )
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
	return src;
}

float4x4 FPU::store( matrix src )
{
	return src;
}



FPU::vector FPU::cross( vector u, vector v )
{
	return flink::cross( u, v );
}

FPU::vector FPU::dot( vector u, vector v )
{
	return vector( flink::dot( u, v ) );
}

FPU::vector FPU::homogenize( vector v )
{
	return vector( flink::homogenize( v ) );
}

FPU::vector FPU::hsum( vector v )
{
	return vector( flink::hsum( v ) );
}

FPU::vector FPU::len( vector v )
{
	return vector( flink::len( v ) );
}

FPU::vector FPU::len_sq( vector v )
{
	return vector( flink::len_sq( v ) );
}

FPU::vector FPU::normalize( vector v )
{
	return flink::normalize( v );
}

#pragma endregion

#pragma region SSE3 Backend

template< class FMA >
typename SSE3< FMA >::vector SSE3< FMA >::set( float s )
{
	return load( float4( s ) );
}

template< class FMA >
typename SSE3< FMA >::vector SSE3< FMA >::set( float x, float y, float z, float w )
{
	return load( float4( x, y, z, w ) );
}

template< class FMA >
typename SSE3< FMA >::vector SSE3< FMA >::load( float4 src )
{
	return _mm_load_ps( src );
}

template< class FMA >
float4 SSE3< FMA >::store( vector src )
{
	float4 result;

	_mm_store_ps( result, src );

	return result;
}



template< class FMA >
typename SSE3< FMA >::matrix SSE3< FMA >::load ( float4x4 src )
{
	matrix result;

	result.row0 = load( src.row0 );
	result.row1 = load( src.row1 );
	result.row2 = load( src.row2 );
	result.row3 = load( src.row3 );

	return result;
}

template< class FMA >
float4x4 SSE3< FMA >::store( matrix src )
{
	float4x4 result;

	result.row0 = store( src.row0 );
	result.row1 = store( src.row1 );
	result.row2 = store( src.row2 );
	result.row3 = store( src.row3 );

	return result;
}



template< class FMA >
typename SSE3< FMA >::vector SSE3< FMA >::cross( vector u, vector v )
{
	vector urot = _mm_shuffle_ps( u, u, _MM_SHUFFLE( 3, 0, 2, 1 ) );
	vector vrot = _mm_shuffle_ps( v, v, _MM_SHUFFLE( 3, 0, 2, 1 ) );

	urot = u * vrot;
	vrot = v * urot;

	urot = urot - vrot;

	return _mm_shuffle_ps( urot, urot, _MM_SHUFFLE( 3, 0, 2, 1 ) );
}

template< class FMA >
typename SSE3< FMA >::vector SSE3< FMA >::dot( vector u, vector v )
{
	return hsum( u * v );
}

template< class FMA >
typename SSE3< FMA >::vector SSE3< FMA >::fma3( vector u, vector v, vector w )
{
	return FMA::fma3( u, v, w );
}

template< class FMA >
typename SSE3< FMA >::vector SSE3< FMA >::homogenize( vector v )
{
	return v / _mm_shuffle_ps( v, v, _MM_SHUFFLE( 3, 3, 3, 3 ) );
}

template< class FMA >
typename SSE3< FMA >::vector SSE3< FMA >::hsum( vector v )
{
	// v == xyzw
	vector rotl1 = _mm_shuffle_ps( v, v, _MM_SHUFFLE( 0, 3, 2, 1 ) ); // yzwx
	vector rotl2 = _mm_shuffle_ps( v, v, _MM_SHUFFLE( 1, 0, 3, 2 ) ); // zwxy
	vector rotl3 = _mm_shuffle_ps( v, v, _MM_SHUFFLE( 2, 1, 0, 3 ) ); // wxyz

	v     = v     + rotl1;
	rotl2 = rotl2 + rotl3;

	return v + rotl2;
}

template< class FMA >
typename SSE3< FMA >::vector SSE3< FMA >::len( vector v )
{
	return _mm_sqrt_ps( len_sq( v ) );
}

template< class FMA >
typename SSE3< FMA >::vector SSE3< FMA >::len_sq( vector v )
{
	return dot( v, v );
}

template< class FMA >
typename SSE3< FMA >::vector SSE3< FMA >::normalize( vector v )
{
	return v / len( v );
}



template< class FMA >
inline typename SSE3< FMA >::vector operator*( typename SSE3< FMA >::vector v, typename SSE3< FMA >::matrix m )
{
	typedef SSE3< FMA >::vector vec;

	vec tmp0 = _mm_shuffle_ps( v, v, _MM_SHUFFLE( 0, 0, 0, 0 ) );
	vec tmp1 = _mm_shuffle_ps( v, v, _MM_SHUFFLE( 1, 1, 1, 1 ) );
	vec tmp2 = _mm_shuffle_ps( v, v, _MM_SHUFFLE( 2, 2, 2, 2 ) );
	vec tmp3 = _mm_shuffle_ps( v, v, _MM_SHUFFLE( 3, 3, 3, 3 ) );

	tmp0 = tmp0 * m.row0;
	tmp1 = tmp1 * m.row1;
	tmp2 = tmp2 * m.row2;
	tmp3 = tmp3 * m.row3;

	tmp0 = tmp0 + tmp1;
	tmp2 = tmp2 + tmp3;

	return tmp0 + tmp2;
}

template< class FMA >
inline typename SSE3< FMA >::matrix operator*( typename SSE3< FMA >::matrix m, typename SSE3< FMA >::matrix n )
{
	typedef SSE3< FMA >::matrix mat;

	mat result;

	result.row0 = m.row0 * n;
	result.row1 = m.row1 * n;
	result.row2 = m.row2 * n;
	result.row3 = m.row3 * n;

	return result;
}

template< class FMA >
inline typename SSE3< FMA >::vector & operator*=( typename SSE3< FMA >::vector & v, typename SSE3< FMA >::matrix m )
{
	v = v * m;
	return v;
}

template< class FMA >
inline typename SSE3< FMA >::matrix & operator*=( typename SSE3< FMA >::matrix & m, typename SSE3< FMA >::matrix n )
{
	m.row0 *= n;
	m.row1 *= n;
	m.row2 *= n;
	m.row3 *= n;

	return m;
}



__m128 SSE_FMA::fma3( __m128 a, __m128 b, __m128 c )
{
	return a * b + c;
}

__m128 AVX_FMA::fma3( __m128 a, __m128 b, __m128 c )
{
	return _mm_fmadd_ps( a, b, c );
}

#pragma endregion

#pragma region float4/float4x4

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



float4x4::float4x4()
{
}

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

} // namespace

#pragma endregion