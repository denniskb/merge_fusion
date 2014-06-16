#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>



namespace kifi {
namespace util {

#pragma region vec3/matrix

__declspec( deprecated )
struct uint4
{
	std::uint32_t x, y, z, w;
};

__declspec( deprecated )
struct tmpfloat4
{
	float x, y, z, w;
};

struct vec3
{
	float x, y, z;

	inline vec3();
	inline explicit vec3( float s );
	inline vec3( float x, float y, float z );
};



struct matrix4x3
{
	// stored in column-major order
	float 
		m00, m10, m20, m30,
		m01, m11, m21, m31,
		m02, m12, m22, m32;

	inline matrix4x3();
	inline matrix4x3
	(
		float m00, float m01, float m02, float m03,
		float m10, float m11, float m12, float m13,
		float m20, float m21, float m22, float m23
	);
};



struct matrix
{
	// stored in column-major order
	float 
		m00, m10, m20, m30,
		m01, m11, m21, m31,
		m02, m12, m22, m32,
		m03, m13, m23, m33;

	inline matrix();
	inline matrix
	(
		float m00, float m01, float m02, float m03,
		float m10, float m11, float m12, float m13,
		float m20, float m21, float m22, float m23,
		float m30, float m31, float m32, float m33
	);
	inline explicit matrix( matrix4x3 const & m );
	// src must be in row-major order
	inline explicit matrix( float const * src );

	inline operator matrix4x3() const;
};



matrix const identity
(
	1.0f, 0.0f, 0.0f, 0.0f,
	0.0f, 1.0f, 0.0f, 0.0f,
	0.0f, 0.0f, 1.0f, 0.0f,
	0.0f, 0.0f, 0.0f, 1.0f
);

#pragma endregion

#pragma region Operators

inline vec3 operator+( vec3 v, float s );
inline vec3 operator-( vec3 v, float s );
inline vec3 operator*( vec3 v, float s );
inline vec3 operator/( vec3 v, float s );

inline vec3 operator+( float s, vec3 v );
inline vec3 operator-( float s, vec3 v );
inline vec3 operator*( float s, vec3 v );
inline vec3 operator/( float s, vec3 v );

inline vec3 operator+( vec3 u, vec3 v );
inline vec3 operator-( vec3 u, vec3 v );
inline vec3 operator*( vec3 u, vec3 v );
inline vec3 operator/( vec3 u, vec3 v );

inline vec3 & operator+=( vec3 & v, float s );
inline vec3 & operator-=( vec3 & v, float s );
inline vec3 & operator*=( vec3 & v, float s );
inline vec3 & operator/=( vec3 & v, float s );

inline vec3 & operator+=( vec3 & u, vec3 v );
inline vec3 & operator-=( vec3 & u, vec3 v );
inline vec3 & operator*=( vec3 & u, vec3 v );
inline vec3 & operator/=( vec3 & u, vec3 v );

inline matrix operator*( matrix m, matrix n );

#pragma endregion

#pragma region Functions

template< typename T >
T               clamp   ( T x, T a, T b );
inline float    dot     ( vec3 u, vec3 v );
template< typename T >
T               lerp    ( T a, T b, T weightB );
inline uint32_t pack    ( uint32_t x, uint32_t y, uint32_t z );
inline bool     powerOf2( unsigned x );
inline void     unpack  ( uint32_t v, uint32_t & outX, uint32_t & outY, uint32_t & outZ );

inline matrix4x3 invert_transform  ( matrix4x3 const & Rt );
inline matrix    perspective_fov_rh( float fovYradians, float aspectWbyH, float nearZdistance, float farZdistance );
// TODO: Try const &
inline vec3      project           ( vec3 v, matrix const & m );    // w == 1, homogenization
inline vec3      transform_point   ( vec3 v, matrix4x3 const & m ); // w == 1, no homogenization
inline vec3      transform_vector  ( vec3 v, matrix4x3 const & m ); // w == 0, no homogenization
inline matrix    transpose         ( matrix const & m );

#pragma endregion

}} // namespace



#pragma region implementation

namespace kifi {
namespace util {

#pragma region vec3/matrix

vec3::vec3() {}
vec3::vec3( float s ) : x( s ), y( s ), z( s ) {}
vec3::vec3( float x, float y, float z ) : x( x ), y( y ), z( z ) {}



matrix::matrix() {}

matrix::matrix
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

matrix::matrix( matrix4x3 const & m )
{
	std::memcpy( & m00, & m.m00, 48 );
	
	m03 = 0.0f;
	m13 = 0.0f;
	m23 = 0.0f;
	m33 = 1.0f;
}

matrix::matrix( float const * src )
{
	std::memcpy( & m00, src, 64 );
	* this = transpose( * this );
}

matrix::operator matrix4x3() const
{
	matrix4x3 result;

	std::memcpy( & result.m00, & m00, 48 );

	return result;
}



matrix4x3::matrix4x3() {}

matrix4x3::matrix4x3
(
	float m00, float m01, float m02,
	float m10, float m11, float m12,
	float m20, float m21, float m22,
	float m30, float m31, float m32
) :
	m00( m00 ), m10( m10 ), m20( m20 ), m30( m30 ),
	m01( m01 ), m11( m11 ), m21( m21 ), m31( m31 ),
	m02( m02 ), m12( m12 ), m22( m22 ), m32( m32 )
{}

#pragma endregion

#pragma region Operators

vec3 operator+( vec3 v, float s )
{
	return vec3
	(
		v.x + s,
		v.y + s,
		v.z + s
	);
}

vec3 operator-( vec3 v, float s )
{
	return vec3
	(
		v.x - s,
		v.y - s,
		v.z - s
	);
}

vec3 operator*( vec3 v, float s )
{
	return vec3
	(
		v.x * s,
		v.y * s,
		v.z * s
	);
}

vec3 operator/( vec3 v, float s )
{
	float const sInv = 1.0f / s;

	return vec3
	(
		v.x * sInv,
		v.y * sInv,
		v.z * sInv
	);
}



vec3 operator+( float s, vec3 v )
{
	return vec3
	(
		s + v.x,
		s + v.y,
		s + v.z
	);
}

vec3 operator-( float s, vec3 v )
{
	return vec3
	(
		s - v.x,
		s - v.y,
		s - v.z
	);
}

vec3 operator*( float s, vec3 v )
{
	return vec3
	(
		s * v.x,
		s * v.y,
		s * v.z
	);
}

vec3 operator/( float s, vec3 v )
{
	return vec3
	(
		s / v.x,
		s / v.y,
		s / v.z
	);
}



vec3 operator+( vec3 u, vec3 v )
{
	return vec3
	(
		u.x + v.x,
		u.y + v.y,
		u.z + v.z
	);
}

vec3 operator-( vec3 u, vec3 v )
{
	return vec3
	(
		u.x - v.x,
		u.y - v.y,
		u.z - v.z
	);
}

vec3 operator*( vec3 u, vec3 v )
{
	return vec3
	(
		u.x * v.x,
		u.y * v.y,
		u.z * v.z
	);
}

vec3 operator/( vec3 u, vec3 v )
{
	return vec3
	(
		u.x / v.x,
		u.y / v.y,
		u.z / v.z
	);
}



vec3 & operator+=( vec3 & v, float s )
{
	v.x += s;
	v.y += s;
	v.z += s;

	return v;
}

vec3 & operator-=( vec3 & v, float s )
{
	v.x -= s;
	v.y -= s;
	v.z -= s;

	return v;
}

vec3 & operator*=( vec3 & v, float s )
{
	v.x *= s;
	v.y *= s;
	v.z *= s;

	return v;
}

vec3 & operator/=( vec3 & v, float s )
{
	float const sInv = 1.0f / s;

	v.x *= sInv;
	v.y *= sInv;
	v.z *= sInv;

	return v;
}



vec3 & operator+=( vec3 & u, vec3 v )
{
	u.x += v.x;
	u.y += v.y;
	u.z += v.z;

	return u;
}

vec3 & operator-=( vec3 & u, vec3 v )
{
	u.x -= v.x;
	u.y -= v.y;
	u.z -= v.z;

	return u;
}

vec3 & operator*=( vec3 & u, vec3 v )
{
	u.x *= v.x;
	u.y *= v.y;
	u.z *= v.z;

	return u;
}

vec3 & operator/=( vec3 & u, vec3 v )
{
	u.x /= v.x;
	u.y /= v.y;
	u.z /= v.z;

	return u;
}



matrix operator*( matrix m, matrix n )
{
	matrix result;

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

#pragma endregion

#pragma region Functions

template< typename T >
T clamp( T x, T a, T b )
{
	return std::max( a, std::min( x, b ) );
}

float dot( vec3 u, vec3 v )
{
	return (u.x * v.x + u.y * v.y) + (u.z * v.z);
}

template< typename T >
T lerp( T a, T b, T weightB )
{
	return a + (b-a) * weightB;
}

uint32_t pack( uint32_t x, uint32_t y, uint32_t z )
{
	return z << 20 | y << 10 | x;
}

bool powerOf2( unsigned x )
{
	return x > 0 && ! (x & (x - 1));
}

void unpack( uint32_t v, uint32_t & outX, uint32_t & outY, uint32_t & outZ )
{
	outX = v & 0x3ff;
	outY = v >> 10 & 0x3ff;
	outZ = v >> 20;
}



matrix4x3 invert_transform( matrix4x3 const & Rt )
{
	matrix R( Rt );
	R.m30 = 0.0f;
	R.m31 = 0.0f;
	R.m32 = 0.0f;

	matrix tInv = identity;
	tInv.m30 = -Rt.m30;
	tInv.m31 = -Rt.m31;
	tInv.m32 = -Rt.m32;

	return tInv * transpose( R );
}

matrix perspective_fov_rh( float fovYradians, float aspectWbyH, float nearZdistance, float farZdistance )
{
	float h = 1.0f / std::tanf( 0.5f * fovYradians );
	float w = h / aspectWbyH;
	float Q = farZdistance / (farZdistance - nearZdistance);
	
	matrix result;
	std::memset( & result.m00, 0, 64 );

	result.m00 = w;
	result.m11 = h;
	result.m22 = -Q;
	result.m32 = -Q * nearZdistance;
	result.m23 = -1.0f;

	return result;
}

vec3 project( vec3 v, matrix const & m )
{
	vec3 result;
	
	result.x = ( v.x * m.m00 + v.y * m.m10 ) + ( v.z * m.m20 + m.m30 );
	result.y = ( v.x * m.m01 + v.y * m.m11 ) + ( v.z * m.m21 + m.m31 );
	result.z = ( v.x * m.m02 + v.y * m.m12 ) + ( v.z * m.m22 + m.m32 );
	float w  = ( v.x * m.m03 + v.y * m.m13 ) + ( v.z * m.m23 + m.m33 );
	
	return result / w;
}

vec3 transform_point( vec3 v, matrix4x3 const & m )
{
	vec3 result;
	
	result.x = ( v.x * m.m00 + v.y * m.m10 ) + ( v.z * m.m20 + m.m30 );
	result.y = ( v.x * m.m01 + v.y * m.m11 ) + ( v.z * m.m21 + m.m31 );
	result.z = ( v.x * m.m02 + v.y * m.m12 ) + ( v.z * m.m22 + m.m32 );
	
	return result;
}

vec3 transform_vector( vec3 v, matrix4x3 const & m )
{
	vec3 result;
	
	result.x = ( v.x * m.m00 + v.y * m.m10 ) + ( v.z * m.m20 );
	result.y = ( v.x * m.m01 + v.y * m.m11 ) + ( v.z * m.m21 );
	result.z = ( v.x * m.m02 + v.y * m.m12 ) + ( v.z * m.m22 );
	
	return result;
}

matrix transpose( matrix const & m )
{
	return matrix
	(
		m.m00, m.m10, m.m20, m.m30,
		m.m01, m.m11, m.m21, m.m31,
		m.m02, m.m12, m.m22, m.m32,
		m.m03, m.m13, m.m23, m.m33
	);
}

#pragma endregion

}} // namespace

#pragma endregion