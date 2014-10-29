//------------------------------------------------------------------------------
// <copyright file="DepthBasics.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

#include "stdafx.h"
#include <cmath>
#include <strsafe.h>
#include "resource.h"
#include "DepthBasics.h"

template< typename T >
inline T minimum( T a, T b )
{
	return a < b ? a : b;
}

template< typename T >
inline T maximum( T a, T b )
{
	return a < b ? b : a;
}

#include <array>

#include <xmmintrin.h> // SSE1



#define kifi_for( n ) for( int i = 0; i < (n); i++ )



namespace kifi {
namespace util {

#pragma region vec< T, N >

template< typename T, int N >
struct vec : public std::array< T, N >
{
	// does not initialize!
	inline vec();
	// initializes to s
	inline explicit vec( T s );
	// only implemented for N == 2/3/4 respectively
	inline vec( T x, T y );
	inline vec( T x, T y, T z );
	inline vec( T x, T y, T z, T w );
	inline vec( vec< T, 3 > xyz, T w );

	// only implemented for N >= 1/2/3/4 respectively
	inline T & x();
	inline T & y();
	inline T & z();
	inline T & w();

	inline T const & x() const;
	inline T const & y() const;
	inline T const & z() const;
	inline T const & w() const;

	inline T & r();
	inline T & g();
	inline T & b();
	inline T & a();

	inline T const & r() const;
	inline T const & g() const;
	inline T const & b() const;
	inline T const & a() const;

	inline vec operator-() const;

	template< typename U >
	inline explicit operator vec< U, N >() const;

	// only implemented for N > 2/3 respectively.
	inline vec< T, 2 > xy() const;
	inline vec< T, 3 > xyz() const;
	inline vec< T, 3 > rgb() const;

	// invariants
	static_assert( N > 0, "a vector must have 1 or more components." );
};

typedef vec< int, 2 > int2;
typedef vec< int, 3 > int3;
typedef vec< int, 4 > int4;

typedef vec< float, 2 > float2;
typedef vec< float, 3 > float3;
typedef vec< float, 4 > float4;

typedef vec< double, 2 > double2;
typedef vec< double, 3 > double3;
typedef vec< double, 4 > double4;

#pragma endregion

#pragma region mat< T, R, C >

template< typename T, int R, int C >
struct mat
{
	std::array< vec< T, R >, C > cols;

	// does not initialize!
	inline mat();
	// initializes to s
	inline explicit mat( T s );
	// only implemented for 4x4 matrices
	inline mat
	(
		T m00, T m01, T m02, T m03,
		T m10, T m11, T m12, T m13,
		T m20, T m21, T m22, T m23,
		T m30, T m31, T m32, T m33
	);
	// src must be in row-major order
	inline explicit mat( T const * src );

	inline T & operator()( int iRow, int iCol );
	inline T const & operator()( int iRow, int iCol ) const;

	template< typename U >
	inline explicit operator mat< U, R, C >() const;

	static inline mat identity();

	// invariants
	static_assert( R > 0 && C > 0, "a matrix must have at least 1 row and 1 column." );
};

typedef mat< float, 4, 4 > float4x4;

#pragma endregion

#pragma region vector/matrix (SSE)

typedef __m128 vector;
struct matrix { vector col0, col1, col2, col3; };

#pragma endregion

#pragma region Operators

// vector op vector
template< typename T, int N >
inline vec< T, N > operator+( vec< T, N > const & a, vec< T, N > const & b );
template< typename T, int N >
inline vec< T, N > operator-( vec< T, N > const & a, vec< T, N > const & b );
template< typename T, int N >
inline vec< T, N > operator*( vec< T, N > const & a, vec< T, N > const & b );
template< typename T, int N >
inline vec< T, N > operator/( vec< T, N > const & a, vec< T, N > const & b );

// vector op= vector
template< typename T, int N >
inline vec< T, N > & operator+=( vec< T, N > & a, vec< T, N > const & b );
template< typename T, int N >
inline vec< T, N > & operator-=( vec< T, N > & a, vec< T, N > const & b );
template< typename T, int N >
inline vec< T, N > & operator*=( vec< T, N > & a, vec< T, N > const & b );
template< typename T, int N >
inline vec< T, N > & operator/=( vec< T, N > & a, vec< T, N > const & b );

// vector op scalar
template< typename T, int N >
inline vec< T, N > operator+( vec< T, N > const & a, T b );
template< typename T, int N >
inline vec< T, N > operator-( vec< T, N > const & a, T b );
template< typename T, int N >
inline vec< T, N > operator*( vec< T, N > const & a, T b );
template< typename T, int N >
inline vec< T, N > operator/( vec< T, N > const & a, T b );

// vector op= scalar
template< typename T, int N >
inline vec< T, N > & operator+=( vec< T, N > & a, T b );
template< typename T, int N >
inline vec< T, N > & operator-=( vec< T, N > & a, T b );
template< typename T, int N >
inline vec< T, N > & operator*=( vec< T, N > & a, T b );
template< typename T, int N >
inline vec< T, N > & operator/=( vec< T, N > & a, T b );

// scalar op vector
template< typename T, int N >
inline vec< T, N > operator+( T a, vec< T, N > const & b );
template< typename T, int N >
inline vec< T, N > operator-( T a, vec< T, N > const & b );
template< typename T, int N >
inline vec< T, N > operator*( T a, vec< T, N > const & b );
template< typename T, int N >
inline vec< T, N > operator/( T a, vec< T, N > const & b );

// matrix op vector/matrix
template< typename T, int R, int C >
inline vec< T, R > operator*( mat< T, R, C > const & m, vec< T, C > const & v );
template< typename T, int R, int C1, int C2 >
inline mat< T, R, C2 > operator*( mat< T, R, C1 > const & m, mat< T, C1, C2 > const & n );

// matrix op vector (SSE)
inline vector operator* ( matrix m, vector v );

#pragma endregion

#pragma region Functions

// vector

template< typename T >
inline vec< T, 3 >           cross         ( vec< T, 3 > const & a, vec< T, 3 > const & b );
// returns ( cross( a.xyz(), b.xyz() ), 0 )
template< typename T >
inline vec< T, 4 >           cross         ( vec< T, 4 > const & a, vec< T, 4 > const & b );
template< typename T, int N >
inline T                     dot           ( vec< T, N > const & a, vec< T, N > const & b );
template< typename T >
inline vec< T, 4 >           homogenize    ( vec< T, 4 > const & a );
template< typename T, int N >
inline bool                  is_nan        ( vec< T, N > const & a );
template< typename T, int N >
inline double                length        ( vec< T, N > const & a );
template< int N >
inline float                 length        ( vec< float, N > const & a );
template< int N >
inline long double           length        ( vec< long double, N > const & a );
template< typename T, int N >
inline T                     length_squared( vec< T, N > const & a );
template< typename T, int N >
inline vec< double, N >      normalize     ( vec< T, N > const & a );
template< int N >
inline vec< float, N >       normalize     ( vec< float, N > const & a );
template< int N >
inline vec< long double, N > normalize     ( vec< long double, N > const & a );

// matrix

// Computes Eigenvectors of a symmetric matrix m.
// Eigenvectors are stored in the columns of 'outEigenVectors'. 'm' and 'outEigenVectors' are allowed to be identical.
template< typename T, int N >
inline void           eigen           ( mat< T, N, N > const & m, vec< T, N > & outEigenValues, mat< T, N, N > & outEigenVectors );
template< typename T >
inline mat< T, 4, 4 > invert_transform( mat< T, 4, 4 > const & tR );
template< typename T, int R, int C >
inline mat< T, C, R > transpose       ( mat< T, R, C > const & m );

// SSE

inline int    all           ( vector v );
inline int    any           ( vector v );
template< int index >	      
inline vector broadcast     ( vector v );
inline vector dot           ( vector u, vector v );
inline vector homogenize    ( vector v );
inline vector length        ( vector v );
inline vector length_squared( vector v );
inline vector load          ( float4 const & src );
inline matrix load          ( float4x4 const & src );
inline vector load          ( float const * src );
inline vector loadu         ( float const * src );
inline vector loadss        ( float src );
inline int    none          ( vector v );
inline vector normalize     ( vector v );
inline vector set           ( float s );
inline vector set           ( float x, float y, float z, float w );
template< int a0, int a1, int b0, int b1 >
inline vector shuffle       ( vector a, vector b );
inline float4 store         ( vector src );
inline void   store         ( float * dst, vector src );
inline void   storeu        ( float * dst, vector src );
inline float  storess       ( vector src );
inline vector zero          ();

// Other

template< typename T >
T               clamp   ( T x, T a, T b );
template< typename T, typename U >
T               lerp    ( T a, T b, U weightB );
inline unsigned	pack    ( unsigned x, unsigned y, unsigned z );
inline bool     powerOf2( int x );
inline void     unpack  ( unsigned v, unsigned & outX, unsigned & outY, unsigned & outZ );

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
#include <functional>



namespace kifi {
namespace util {

#pragma region vec< T, N >

// Constructors

template< typename T, int N >
vec< T, N >::vec()
{	
}

template< typename T, int N >
vec< T, N >::vec( T s )
{
	fill( s );		
}

template< typename T, int N >
vec< T, N >::vec( T x, T y )
{
	static_assert( 2 == N, "this constructor is only implemented for 2-component vectors" );

	(*this)[ 0 ] = x;
	(*this)[ 1 ] = y;
}

template< typename T, int N >
vec< T, N >::vec( T x, T y, T z )
{
	static_assert( 3 == N, "this constructor is only implemented for 3-component vectors" );

	(*this)[ 0 ] = x;
	(*this)[ 1 ] = y;
	(*this)[ 2 ] = z;
}

template< typename T, int N >
vec< T, N >::vec( T x, T y, T z, T w )
{
	static_assert( 4 == N, "this constructor is only implemented for 4-component vectors" );

	(*this)[ 0 ] = x;
	(*this)[ 1 ] = y;
	(*this)[ 2 ] = z;
	(*this)[ 3 ] = w;
}

template< typename T, int N >
vec< T, N >::vec( vec< T, 3 > xyz, T w )
{
	static_assert( 4 == N, "this constructor is only implemented for 4-component vectors" );

	(*this)[ 0 ] = xyz.x();
	(*this)[ 1 ] = xyz.y();
	(*this)[ 2 ] = xyz.z();
	(*this)[ 3 ] = w;
}

// Acessors

template< typename T, int N >
T & vec< T, N >::x()
{
	static_assert( 1 <= N, "can't access x-component of a vector with less than 1 component");

	return (*this)[ 0 ];
}

template< typename T, int N >
T & vec< T, N >::y()
{
	static_assert( 2 <= N, "can't access y-component of a vector with less than 2 components");

	return (*this)[ 1 ];
}

template< typename T, int N >
T & vec< T, N >::z()
{
	static_assert( 3 <= N, "can't access z-component of a vector with less than 3 components");

	return (*this)[ 2 ];
}

template< typename T, int N >
T & vec< T, N >::w()
{
	static_assert( 4 <= N, "can't access w-component of a vector with less than 4 components");

	return (*this)[ 3 ];
}



template< typename T, int N >
T const & vec< T, N >::x() const
{
	static_assert( 1 <= N, "can't access x-component of a vector with less than 1 component");

	return (*this)[ 0 ];
}

template< typename T, int N >
T const & vec< T, N >::y() const
{
	static_assert( 2 <= N, "can't access y-component of a vector with less than 2 components");

	return (*this)[ 1 ];
}

template< typename T, int N >
T const & vec< T, N >::z() const
{
	static_assert( 3 <= N, "can't access z-component of a vector with less than 3 components");

	return (*this)[ 2 ];
}

template< typename T, int N >
T const & vec< T, N >::w() const
{
	static_assert( 4 <= N, "can't access w-component of a vector with less than 4 components");

	return (*this)[ 3 ];
}



template< typename T, int N >
T & vec< T, N >::r()
{
	return x();
}

template< typename T, int N >
T & vec< T, N >::g()
{
	return y();
}

template< typename T, int N >
T & vec< T, N >::b()
{
	return z();
}

template< typename T, int N >
T & vec< T, N >::a()
{
	return w();
}



template< typename T, int N >
T const & vec< T, N >::r() const
{
	return x();
}

template< typename T, int N >
T const & vec< T, N >::g() const
{
	return y();
}

template< typename T, int N >
T const & vec< T, N >::b() const
{
	return z();
}

template< typename T, int N >
T const & vec< T, N >::a() const
{
	return w();
}

// Methods

template< typename T, int N >
vec< T, N > vec< T, N >::operator-() const
{
	vec< T, N > result;

	kifi_for( N )
		result[ i ] = -(*this)[ i ];

	return result;
}

template< typename T, int N >
template< typename U >
vec< T, N >::operator vec< U, N >() const
{
	vec< U, N > result;

	kifi_for( N )
		result[ i ] = (U) (*this)[ i ];

	return result;
}

template< typename T, int N >
vec< T, 2 > vec< T, N >::xy() const
{
	static_assert( N > 2, "this method is only implemented for (3+)-component vectors" );

	return vec< T, 2 >( x(), y() );
}

template< typename T, int N >
vec< T, 3 > vec< T, N >::xyz() const
{
	static_assert( N > 3, "this method is only implemented for (4+)-component vectors" );

	return vec< T, 3 >( x(), y(), z() );
}

template< typename T, int N >
vec< T, 3 > vec< T, N >::rgb() const
{
	return xyz();
}

#pragma endregion

#pragma region mat< T, R, C >

// Constructors

template< typename T, int R, int C >
mat< T, R, C >::mat()
{
}

template< typename T, int R, int C >
mat< T, R, C >::mat( T s )
{
	cols.fill( vec< T, R >( s ) );
}

template< typename T, int R, int C >
mat< T, R, C >::mat
(
	T m00, T m01, T m02, T m03,
	T m10, T m11, T m12, T m13,
	T m20, T m21, T m22, T m23,
	T m30, T m31, T m32, T m33
)
{
	static_assert( 4 == R && 4 == C, "this constructor is only implemented for 4x4 matrices" );

	cols[ 0 ] = vec< T, R >( m00, m10, m20, m30 );
	cols[ 1 ] = vec< T, R >( m01, m11, m21, m31 );
	cols[ 2 ] = vec< T, R >( m02, m12, m22, m32 );
	cols[ 3 ] = vec< T, R >( m03, m13, m23, m33 );
}
	
template< typename T, int R, int C >
mat< T, R, C >::mat( T const * src )
{
	for( int col = 0; col < C; col++ )
		for( int row = 0; row < R; row++ )
			(*this)( row, col ) = src[ col + row * C ];
}

// Accessors

template< typename T, int R, int C >
T & mat< T, R, C >::operator()( int iRow, int iCol )
{
	assert( iRow >= 0 && iRow < R &&
			iCol >= 0 && iCol < C );

	return cols[ iCol ][ iRow ];
}

template< typename T, int R, int C >
T const & mat< T, R, C >::operator()( int iRow, int iCol ) const
{
	assert( iRow >= 0 && iRow < R &&
			iCol >= 0 && iCol < C );

	return cols[ iCol ][ iRow ];
}

// Methods

template< typename T, int R, int C >
template< typename U >
mat< T, R, C >::operator mat< U, R, C >() const
{
	mat< U, R, C > result;

	for( int col = 0; col < C; col++ )
		for( int row = 0; row < R; row++ )
			result( row, col ) = (U) (*this)( row, col );

	return result;
}

// static 
template< typename T, int R, int C >
mat< T, R, C > mat< T, R, C >::identity()
{
	mat< T, R, C > result( (T) 0 );

	kifi_for( std::min( R, C ) )
		result( i, i ) = (T) 1;

	return result;
}

#pragma endregion

#pragma region Operators

// vector op vector

template< typename T, int N >
vec< T, N > operator+( vec< T, N > const & a, vec< T, N > const & b )
{
	vec< T, N > result;

	kifi_for( N )
		result[ i ] = a[ i ] + b[ i ];

	return result;
}

template< typename T, int N >
vec< T, N > operator-( vec< T, N > const & a, vec< T, N > const & b )
{
	vec< T, N > result;

	kifi_for( N )
		result[ i ] = a[ i ] - b[ i ];

	return result;
}

template< typename T, int N >
vec< T, N > operator*( vec< T, N > const & a, vec< T, N > const & b )
{
	vec< T, N > result;

	kifi_for( N )
		result[ i ] = a[ i ] * b[ i ];

	return result;
}

template< typename T, int N >
vec< T, N > operator/( vec< T, N > const & a, vec< T, N > const & b )
{
	vec< T, N > result;

	kifi_for( N )
		result[ i ] = a[ i ] / b[ i ];

	return result;
}

// vector op= vector

template< typename T, int N >
vec< T, N > & operator+=( vec< T, N > & a, vec< T, N > const & b )
{
	kifi_for( N )
		a[ i ] += b[ i ];

	return a;
}

template< typename T, int N >
vec< T, N > & operator-=( vec< T, N > & a, vec< T, N > const & b )
{
	kifi_for( N )
		a[ i ] -= b[ i ];

	return a;
}

template< typename T, int N >
vec< T, N > & operator*=( vec< T, N > & a, vec< T, N > const & b )
{
	kifi_for( N )
		a[ i ] *= b[ i ];

	return a;
}

template< typename T, int N >
vec< T, N > & operator/=( vec< T, N > & a, vec< T, N > const & b )
{
	kifi_for( N )
		a[ i ] /= b[ i ];

	return a;
}

// vector op scalar

template< typename T, int N >
vec< T, N > operator+( vec< T, N > const & a, T b )
{
	vec< T, N > result;

	kifi_for( N )
		result[ i ] = a[ i ] + b;

	return result;
}

template< typename T, int N >
vec< T, N > operator-( vec< T, N > const & a, T b )
{
	vec< T, N > result;

	kifi_for( N )
		result[ i ] = a[ i ] - b;

	return result;
}

template< typename T, int N >
vec< T, N > operator*( vec< T, N > const & a, T b )
{
	vec< T, N > result;

	kifi_for( N )
		result[ i ] = a[ i ] * b;

	return result;
}

template< typename T, int N >
vec< T, N > operator/( vec< T, N > const & a, T b )
{
	vec< T, N > result;

	kifi_for( N )
		result[ i ] = a[ i ] / b;

	return result;
}

template< int N >
vec< float, N > operator/( vec< float, N > const & a, float b )
{
	return a * (1.0f / b);
}

template< int N >
vec< double, N > operator/( vec< double, N > const & a, double b )
{
	return a * (1.0 / b);
}

template< int N >
vec< long double, N > operator/( vec< long double, N > const & a, long double b )
{
	return a * (1.0 / b);
}

// vector op= scalar

template< typename T, int N >
vec< T, N > & operator+=( vec< T, N > & a, T b )
{
	kifi_for( N )
		a[ i ] += b;

	return a;
}

template< typename T, int N >
vec< T, N > & operator-=( vec< T, N > & a, T b )
{
	kifi_for( N )
		a[ i ] -= b;

	return a;
}

template< typename T, int N >
vec< T, N > & operator*=( vec< T, N > & a, T b )
{
	kifi_for( N )
		a[ i ] *= b;

	return a;
}

template< typename T, int N >
vec< T, N > & operator/=( vec< T, N > & a, T b )
{
	kifi_for( N )
		a[ i ] /= b;

	return a;
}

template< int N >
vec< float, N > & operator/=( vec< float, N > & a, float b )
{
	return a *= (1.0f / b);
}

template< int N >
vec< double, N > & operator/=( vec< double, N > & a, double b )
{
	return a *= (1.0 / b);
}

template< int N >
vec< long double, N > & operator/=( vec< long double, N > & a, long double b )
{
	return a *= (1.0 / b);
}

// scalar op vector

template< typename T, int N >
inline vec< T, N > operator+( T a, vec< T, N > const & b )
{
	return b + a;
}

template< typename T, int N >
inline vec< T, N > operator-( T a, vec< T, N > const & b )
{
	vec< T, N > result;

	kifi_for( N )
		result[ i ] = a - b[ i ];

	return result;
}

template< typename T, int N >
inline vec< T, N > operator*( T a, vec< T, N > const & b )
{
	return b * a;
}

template< typename T, int N >
inline vec< T, N > operator/( T a, vec< T, N > const & b )
{
	vec< T, N > result;

	kifi_for( N )
		result[ i ] = a / b[ i ];

	return result;
}

// matrix op vector/matrix

template< typename T, int R, int C >
vec< T, R > operator*( mat< T, R, C > const & m, vec< T, C > const & v )
{
	vec< T, R > result( (T) 0 );

	kifi_for( C )
		result += m.cols[ i ] * v[ i ];

	return result;
}

template< typename T, int R >
vec< T, R > operator*( mat< T, R, 4 > const & m, vec< T, 4 > const & v )
{
	return ( m.cols[ 0 ] * v.x() + m.cols[ 1 ] * v.y() ) + ( m.cols[ 2 ] * v.z() + m.cols[ 3 ] * v.w() );
}

template< typename T, int R, int C1, int C2 >
mat< T, R, C2 > operator*( mat< T, R, C1 > const & m, mat< T, C1, C2 > const & n )
{
	mat< T, R, C2 > result;

	kifi_for( C2 )
		result.cols[ i ] = m * n.cols[ i ];

	return result;
}

// matrix op vector (SSE)

vector operator*( matrix m, vector v )
{
	vector vx = broadcast< 0 >( v );
	vector vy = broadcast< 1 >( v );
	vector vz = broadcast< 2 >( v );
	vector vw = broadcast< 3 >( v );

	vx *= m.col0;
	vy *= m.col1;
	vz *= m.col2;
	vw *= m.col3;

	vx += vy;
	vz += vw;

	return vx + vz;
}

#pragma endregion

#pragma region Functions

// vector

template< typename T >
inline vec< T, 3 > cross( vec< T, 3 > const & a, vec< T, 3 > const & b )
{
	return vec< T, 3 >
	(
		a.y() * b.z() - a.z() * b.y(),
		a.z() * b.x() - a.x() * b.z(),
		a.x() * b.y() - a.y() * b.x()
	);
}

template< typename T >
inline vec< T, 4 > cross( vec< T, 4 > const & a, vec< T, 4 > const & b )
{
	return vec< T, 4 >
	(
		a.y() * b.z() - a.z() * b.y(),
		a.z() * b.x() - a.x() * b.z(),
		a.x() * b.y() - a.y() * b.x(),
		T()
	);
}

template< typename T, int N >
inline T dot( vec< T, N > const & a, vec< T, N > const & b )
{
	T result = T();

	kifi_for( N )
		result += a[ i ] * b[ i ];

	return result;
}

template< typename T >
inline T dot( vec< T, 4 > const & a, vec< T, 4 > const & b )
{
	return ( a.x() * b.x() + a.y() * b.y() ) + ( a.z() * b.z() + a.w() * b.w() );
}

template< typename T >
inline vec< T, 4 > homogenize( vec< T, 4 > const & a )
{
	return a / a.w();
}

template< typename T, int N >
bool isnan( vec< T, N > const & a )
{
    bool result = false;

    kifi_for( N )
        result |= std::isnan( a[ i ] );

    return result;
}

template< typename T, int N >
inline double length( vec< T, N > const & a )
{
	return std::sqrt( length_squared( a ) );
}

template< int N >
inline float length( vec< float, N > const & a )
{
	return std::sqrt( length_squared( a ) );
}

template< int N >
inline long double length( vec< long double, N > const & a )
{
	return std::sqrt( length_squared( a ) );
}

template< typename T, int N >
inline T length_squared( vec< T, N > const & a )
{
	return dot( a, a );
}

template< typename T, int N >
inline vec< double, N > normalize( vec< T, N > const & a )
{
	return (vec< double, N >) a / length( a );
}

template< int N >
inline vec< float, N > normalize( vec< float, N > const & a )
{
	return a / length( a );
}

template< int N >
inline vec< long double, N > normalize( vec< long double, N > const & a )
{
	return a / length( a );
}

// matrix

template< typename T, int N >
void eigen( mat< T, N, N > const & m, vec< T, N > & outEigenValues, mat< T, N, N > & outEigenVectors )
{
	vec< T, N > tmp;
	outEigenVectors = m;

	// TODO: Replace copy-righted code.
#pragma region tred2/tqli implementation

	static int const MAX_ITERS = 30;
	auto SIGN = [] ( T a, T b ) { return b < ((T)0) ? -std::abs( a ) : std::abs( a ); };

	auto tred2 = [] ( mat< T, N, N > & a, vec< T, N > & d, vec< T, N > & e )
	{
		int l, k, j, i;
        T scale, hh, h, g, f;

        for (i = N - 1; i >= 1; i--)
        {
            l = i - 1;
            h = scale = ((T)0);
            if (l > 0)
            {
                for (k = 0; k <= l; k++)
                    scale += std::abs(a(i,k));
                if (scale == ((T)0))
                    e[i] = a(i,l);
                else
                {
                    for (k = 0; k <= l; k++)
                    {
                        a(i,k) /= scale;
                        h += a(i,k) * a(i,k);
                    }
                    f = a(i,l);
                    g = f > 0 ? -std::sqrt(h) : std::sqrt(h);
                    e[i] = scale * g;
                    h -= f * g;
                    a(i,l) = f - g;
                    f = ((T)0);
                    for (j = 0; j <= l; j++)
                    {
                        /* Next statement can be omitted if eigenvectors not wanted */
                        a(j,i) = a(i,j) / h;
                        g = ((T)0);
                        for (k = 0; k <= j; k++)
                            g += a(j,k) * a(i,k);
                        for (k = j + 1; k <= l; k++)
                            g += a(k,j) * a(i,k);
                        e[j] = g / h;
                        f += e[j] * a(i,j);
                    }
                    hh = f / (h + h);
                    for (j = 0; j <= l; j++)
                    {
                        f = a(i,j);
                        e[j] = g = e[j] - hh * f;
                        for (k = 0; k <= j; k++)
                            a(j,k) -= (f * e[k] + g * a(i,k));
                    }
                }
            }
            else
                e[i] = a(i,l);
            d[i] = h;
        }
        /* Next statement can be omitted if eigenvectors not wanted */
        d[0] = ((T)0);
        e[0] = ((T)0);
        /* Contents of this loop can be omitted if eigenvectors not 
                wanted except for statement d[i]=a(i,i); */
        for (i = 0; i < N; i++)
        {
            l = i - 1;
            if (d[i] != ((T)0))
            {
                for (j = 0; j <= l; j++)
                {
                    g = ((T)0);
                    for (k = 0; k <= l; k++)
                        g += a(i,k) * a(k,j);
                    for (k = 0; k <= l; k++)
                        a(k,j) -= g * a(k,i);
                }
            }
            d[i] = a(i,i);
            a(i,i) = ((T)1);
            for (j = 0; j <= l; j++) a(j,i) = a(i,j) = ((T)0);
        }
	};

	auto tqli =	[ & SIGN ] ( vec< T, N > & d, vec< T, N > & e, mat< T, N, N > & z )
	{
		int m,l,iter,i,k;  
        T s,r,p,g,f,dd,c,b;  
      
        for (i=1;i<N;i++) e[i-1]=e[i];
        e[N - 1] = ((T)0);
        for (l = 0; l < N; l++)
        {  
            iter=0;  
            do {
                for (m = l; m < N - 1; m++)
                {  
                    dd=std::abs(d[m])+std::abs(d[m+1]);  
                    if (std::abs(e[m])+dd == dd) break;  
                }  
                if (m != l) {  
                    if (iter++ == MAX_ITERS) return; /* Too many iterations in TQLI */  
                    g=(d[l+1]-d[l])/(((T)2)*e[l]);  
                    r=std::sqrt((g*g)+((T)1));  
                    g=d[m]-d[l]+e[l]/(g+SIGN(r,g));  
                    s=c=((T)1);  
                    p=((T)0);  
                    for (i=m-1;i>=l;i--) {  
                        f=s*e[i];  
                        b=c*e[i];  
                        if (std::abs(f) >= std::abs(g)) {  
                            c=g/f;  
                            r=std::sqrt((c*c)+((T)1));  
                            e[i+1]=f*r;  
                            c *= (s=((T)1)/r);  
                        } else {  
                            s=f/g;  
                            r=std::sqrt((s*s)+((T)1));  
                            e[i+1]=g*r;  
                            s *= (c=((T)1)/r);  
                        }  
                        g=d[i+1]-p;  
                        r=(d[i]-g)*s+((T)2)*c*b;  
                        p=s*r;  
                        d[i+1]=g+p;  
                        g=c*r-b;  
                        /* Next loop can be omitted if eigenvectors not wanted */
                        for (k = 0; k < N; k++)
                        {  
                            f=z(k,i+1);  
                            z(k,i+1)=s*z(k,i)+c*f;  
                            z(k,i)=c*z(k,i)-s*f;  
                        }  
                    }  
                    d[l]=d[l]-p;  
                    e[l]=g;  
                    e[m]=((T)0);  
                }  
            } while (m != l);  
        } 
	};

#pragma endregion

	tred2( outEigenVectors, outEigenValues, tmp );
	tqli ( outEigenValues, tmp, outEigenVectors );
}

template< typename T >
mat< T, 4, 4 > invert_transform( mat< T, 4, 4 > const & tR )
{
	// isolate and invert translation via negation
	auto tInv = -tR.cols[ 3 ];
	tInv.w() = 1.0f;
	
	// isolate rotation
	mat< T, 4, 4 > R = tR;
	R.cols[ 3 ] = vec< T, 4 >( 0.0f, 0.0f, 0.0f, 1.0f );

	// invert rotation via transposition
	mat< T, 4, 4 > RInv = transpose( R );

	// multiply the two
	RInv.cols[ 3 ] = RInv * tInv;
	return RInv;
}

template< typename T, int R, int C >
mat< T, C, R > transpose( mat< T, R, C > const & m )
{
	mat< T, C, R > result;

	for( int col = 0; col < R; col++ )
		for( int row = 0; row < C; row++ )
			result( row, col ) = m( col, row );

	return result;
}

// SSE

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

vector homogenize( vector v )
{
	return v / broadcast< 3 >( v );
}

vector length( vector v )
{
	return _mm_sqrt_ps( length_squared( v ) );
}

vector length_squared( vector v )
{
	return dot( v, v );
}

vector load( float4 const & src )
{
	return loadu( reinterpret_cast< float const * >( & src ) );
}

matrix load( float4x4 const & src )
{
	matrix result;

	result.col0 = load( src.cols[ 0 ] );
	result.col1 = load( src.cols[ 1 ] );
	result.col2 = load( src.cols[ 2 ] );
	result.col3 = load( src.cols[ 3 ] );

	return result;
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

vector normalize( vector v )
{
	return v / length( v );
}

vector set( float s )
{
	return _mm_set1_ps( s );
}

vector set( float x, float y, float z, float w )
{
	return _mm_set_ps( w, z, y, x );
}

template< int a0, int a1, int b0, int b1 >
vector shuffle( vector a, vector b )
{
	return _mm_shuffle_ps( a, b, _MM_SHUFFLE( b1, b0, a1, a0 ) );
}

float4 store( vector src )
{
	float4 result;

	storeu( reinterpret_cast< float * >( & result ), src );

	return result;
}

void store( float * dst, vector src )
{
	_mm_store_ps( dst, src );
}

void storeu( float * dst, vector src )
{
	_mm_storeu_ps( dst, src );
}

float storess( vector src )
{
	float result;
	_mm_store_ss( & result, src );
	return result;
}

vector zero()
{
	return _mm_setzero_ps();
}

// Other

template< typename T >
T clamp( T x, T a, T b )
{
	return maximum( a, minimum( x, b ) );
}

template< typename T, typename U >
T lerp( T a, T b, U weightB )
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

using namespace kifi::util;

/// <summary>
/// Entry point for the application
/// </summary>
/// <param name="hInstance">handle to the application instance</param>
/// <param name="hPrevInstance">always 0</param>
/// <param name="lpCmdLine">command line arguments</param>
/// <param name="nCmdShow">whether to display minimized, maximized, or normally</param>
/// <returns>status</returns>
int APIENTRY wWinMain(
	_In_ HINSTANCE hInstance,
    _In_opt_ HINSTANCE hPrevInstance,
    _In_ LPWSTR lpCmdLine,
    _In_ int nShowCmd
)
{
    UNREFERENCED_PARAMETER(hPrevInstance);
    UNREFERENCED_PARAMETER(lpCmdLine);

    CDepthBasics application;
    application.Run(hInstance, nShowCmd);
}

/// <summary>
/// Constructor
/// </summary>
CDepthBasics::CDepthBasics() :
    m_hWnd(NULL),
    m_nStartTime(0),
    m_nLastCounter(0),
    m_nFramesSinceUpdate(0),
    m_fFreq(0),
    m_nNextStatusTime(0LL),
    m_bSaveScreenshot(false),
    m_pKinectSensor(NULL),
    m_pDepthFrameReader(NULL),
    m_pD2DFactory(NULL),
    m_pDrawDepth(NULL),
    m_pDepthRGBX(NULL),

	m_bRecording( false ),
	m_pDepthStream( nullptr )
{
    LARGE_INTEGER qpf = {0};
    if (QueryPerformanceFrequency(&qpf))
    {
        m_fFreq = double(qpf.QuadPart);
    }

    // create heap storage for depth pixel data in RGBX format
    m_pDepthRGBX = new RGBQUAD[cDepthWidth * cDepthHeight];
}
  

/// <summary>
/// Destructor
/// </summary>
CDepthBasics::~CDepthBasics()
{
    // clean up Direct2D renderer
    if (m_pDrawDepth)
    {
        delete m_pDrawDepth;
        m_pDrawDepth = NULL;
    }

    if (m_pDepthRGBX)
    {
        delete [] m_pDepthRGBX;
        m_pDepthRGBX = NULL;
    }

    // clean up Direct2D
    SafeRelease(m_pD2DFactory);

    // done with depth frame reader
    SafeRelease(m_pDepthFrameReader);

    // close the Kinect Sensor
    if (m_pKinectSensor)
    {
        m_pKinectSensor->Close();
    }

    SafeRelease(m_pKinectSensor);
}

/// <summary>
/// Creates the main window and begins processing
/// </summary>
/// <param name="hInstance">handle to the application instance</param>
/// <param name="nCmdShow">whether to display minimized, maximized, or normally</param>
int CDepthBasics::Run(HINSTANCE hInstance, int nCmdShow)
{
    MSG       msg = {0};
    WNDCLASS  wc;

    // Dialog custom window class
    ZeroMemory(&wc, sizeof(wc));
    wc.style         = CS_HREDRAW | CS_VREDRAW;
    wc.cbWndExtra    = DLGWINDOWEXTRA;
    wc.hCursor       = LoadCursorW(NULL, IDC_ARROW);
    wc.hIcon         = LoadIconW(hInstance, MAKEINTRESOURCE(IDI_APP));
    wc.lpfnWndProc   = DefDlgProcW;
    wc.lpszClassName = L"DepthBasicsAppDlgWndClass";

    if (!RegisterClassW(&wc))
    {
        return 0;
    }

    // Create main application window
    HWND hWndApp = CreateDialogParamW(
        NULL,
        MAKEINTRESOURCE(IDD_APP),
        NULL,
        (DLGPROC)CDepthBasics::MessageRouter, 
        reinterpret_cast<LPARAM>(this));

    // Show window
    ShowWindow(hWndApp, nCmdShow);

    // Main message loop
    while (WM_QUIT != msg.message)
    {
        Update();

        while (PeekMessageW(&msg, NULL, 0, 0, PM_REMOVE))
        {
            // If a dialog message will be taken care of by the dialog proc
            if (hWndApp && IsDialogMessageW(hWndApp, &msg))
            {
                continue;
            }

            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    }

    return static_cast<int>(msg.wParam);
}

/// <summary>
/// Main processing function
/// </summary>
void CDepthBasics::Update()
{
    if (!m_pDepthFrameReader)
    {
        return;
    }

    IDepthFrame* pDepthFrame = NULL;

    HRESULT hr = m_pDepthFrameReader->AcquireLatestFrame(&pDepthFrame);

    if (SUCCEEDED(hr))
    {
        INT64 nTime = 0;
        IFrameDescription* pFrameDescription = NULL;
        int nWidth = 0;
        int nHeight = 0;
        USHORT nDepthMinReliableDistance = 0;
        USHORT nDepthMaxDistance = 0;
        UINT16 *pBuffer = NULL;

        hr = pDepthFrame->get_RelativeTime(&nTime);

        if (SUCCEEDED(hr))
        {
            hr = pDepthFrame->get_FrameDescription(&pFrameDescription);
        }

        if (SUCCEEDED(hr))
        {
            hr = pFrameDescription->get_Width(&nWidth);
        }

        if (SUCCEEDED(hr))
        {
            hr = pFrameDescription->get_Height(&nHeight);
        }

        if (SUCCEEDED(hr))
        {
            hr = pDepthFrame->get_DepthMinReliableDistance(&nDepthMinReliableDistance);
        }

        if (SUCCEEDED(hr))
        {
			// In order to see the full range of depth (including the less reliable far field depth)
			// we are setting nDepthMaxDistance to the extreme potential depth threshold
			nDepthMaxDistance = USHRT_MAX;

			// Note:  If you wish to filter by reliable depth distance, uncomment the following line.
            //// hr = pDepthFrame->get_DepthMaxReliableDistance(&nDepthMaxDistance);
        }

        if (SUCCEEDED(hr))
        {
			UINT nBufferSize = 0;
            hr = pDepthFrame->AccessUnderlyingBuffer(&nBufferSize, &pBuffer);            
        }

        if (SUCCEEDED(hr))
        {
            ProcessDepth(nTime, pBuffer, nWidth, nHeight, nDepthMinReliableDistance, nDepthMaxDistance);
        }

        SafeRelease(pFrameDescription);
    }

    SafeRelease(pDepthFrame);
}

/// <summary>
/// Handles window messages, passes most to the class instance to handle
/// </summary>
/// <param name="hWnd">window message is for</param>
/// <param name="uMsg">message</param>
/// <param name="wParam">message data</param>
/// <param name="lParam">additional message data</param>
/// <returns>result of message processing</returns>
LRESULT CALLBACK CDepthBasics::MessageRouter(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    CDepthBasics* pThis = NULL;
    
    if (WM_INITDIALOG == uMsg)
    {
        pThis = reinterpret_cast<CDepthBasics*>(lParam);
        SetWindowLongPtr(hWnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(pThis));
    }
    else
    {
        pThis = reinterpret_cast<CDepthBasics*>(::GetWindowLongPtr(hWnd, GWLP_USERDATA));
    }

    if (pThis)
    {
        return pThis->DlgProc(hWnd, uMsg, wParam, lParam);
    }

    return 0;
}

/// <summary>
/// Handle windows messages for the class instance
/// </summary>
/// <param name="hWnd">window message is for</param>
/// <param name="uMsg">message</param>
/// <param name="wParam">message data</param>
/// <param name="lParam">additional message data</param>
/// <returns>result of message processing</returns>
LRESULT CALLBACK CDepthBasics::DlgProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    UNREFERENCED_PARAMETER(wParam);
    UNREFERENCED_PARAMETER(lParam);

    switch (message)
    {
        case WM_INITDIALOG:
        {
            // Bind application window handle
            m_hWnd = hWnd;

            // Init Direct2D
            D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, &m_pD2DFactory);

            // Create and initialize a new Direct2D image renderer (take a look at ImageRenderer.h)
            // We'll use this to draw the data we receive from the Kinect to the screen
            m_pDrawDepth = new ImageRenderer();
            HRESULT hr = m_pDrawDepth->Initialize(GetDlgItem(m_hWnd, IDC_VIDEOVIEW), m_pD2DFactory, cDepthWidth, cDepthHeight, cDepthWidth * sizeof(RGBQUAD)); 
            if (FAILED(hr))
            {
                SetStatusMessage(L"Failed to initialize the Direct2D draw device.", 10000, true);
            }

            // Get and initialize the default Kinect sensor
            InitializeDefaultSensor();
        }
        break;

        // If the titlebar X is clicked, destroy app
        case WM_CLOSE:
            DestroyWindow(hWnd);
            break;

        case WM_DESTROY:
            // Quit the main message pump
            PostQuitMessage(0);
            break;

        // Handle button press
        case WM_COMMAND:
            // If it was for the screenshot control and a button clicked event, save a screenshot next frame 
            if (IDC_BUTTON_SCREENSHOT == LOWORD(wParam) && BN_CLICKED == HIWORD(wParam))
            {
                m_bSaveScreenshot = true;
            }
            break;
    }

    return FALSE;
}

/// <summary>
/// Initializes the default Kinect sensor
/// </summary>
/// <returns>indicates success or failure</returns>
HRESULT CDepthBasics::InitializeDefaultSensor()
{
    HRESULT hr;

    hr = GetDefaultKinectSensor(&m_pKinectSensor);
    if (FAILED(hr))
    {
        return hr;
    }

    if (m_pKinectSensor)
    {
        // Initialize the Kinect and get the depth reader
        IDepthFrameSource* pDepthFrameSource = NULL;

        hr = m_pKinectSensor->Open();

        if (SUCCEEDED(hr))
        {
            hr = m_pKinectSensor->get_DepthFrameSource(&pDepthFrameSource);
        }

        if (SUCCEEDED(hr))
        {
            hr = pDepthFrameSource->OpenReader(&m_pDepthFrameReader);
        }

        SafeRelease(pDepthFrameSource);
    }

    if (!m_pKinectSensor || FAILED(hr))
    {
        SetStatusMessage(L"No ready Kinect found!", 10000, true);
        return E_FAIL;
    }

    return hr;
}

static void intensity_to_rgb( unsigned mm, unsigned minmm, unsigned maxmm, unsigned char & r, unsigned char & g, unsigned char & b )
{
	float n = mm / (float) maxmm;

	float x, y, z;

	x = (n - 0.5f) * 6.0f; x = clamp( x, 0.0f, 1.0f );
	y = minimum( 1.0f, n * 3.0f ) - maximum( 0.0f, n * 3.0f - 2.0f );
	z = 1.0f - (n - 0.333f) * 6.0f; z = clamp( z, 0.0f, 1.0f ); z *= (mm != 0);

	r = (unsigned char) (x * 255.0f);
	g = (unsigned char) (y * 255.0f);
	b = (unsigned char) (z * 255.0f);
}

float fastexp( float x )
{
  x = 1.0f + x / 256.0f;

  return ( (x * x) * (x * x) ) * ( (x * x) * (x * x) );
}

static unsigned short s_buffer[ 512 * 424 ];
static unsigned short s_buffer2[ 512 * 424 ];
static float3 s_pcl[ 512 * 424 ];

/// <summary>
/// Handle new depth data
/// <param name="nTime">timestamp of frame</param>
/// <param name="pBuffer">pointer to frame data</param>
/// <param name="nWidth">width (in pixels) of input image data</param>
/// <param name="nHeight">height (in pixels) of input image data</param>
/// <param name="nMinDepth">minimum reliable depth</param>
/// <param name="nMaxDepth">maximum reliable depth</param>
/// </summary>
void CDepthBasics::ProcessDepth(INT64 nTime, UINT16* pBuffer, int nWidth, int nHeight, USHORT nMinDepth, USHORT nMaxDepth)
{
	nMaxDepth = 3000;

    // Make sure we've received valid data
    if (m_pDepthRGBX && pBuffer && (nWidth == cDepthWidth) && (nHeight == cDepthHeight))
    {
		// Mirror depth map
		for( int idst = 0, res = nWidth * nHeight; idst < res; idst++ )
		{
			int x = idst % nWidth;
			int isrc = idst + nWidth - 1 - 2 * x;

			int tmp = pBuffer[ isrc ];

			tmp *= (tmp >= nMinDepth) * (tmp <= nMaxDepth);
				
			s_buffer[ idst ] = tmp;
		}

		// Write to file
		if( m_bRecording )
		{
			float matrix[ 16 ];
			fwrite( matrix, 4, 16, m_pDepthStream );
			fwrite( s_buffer, 2, nWidth * nHeight, m_pDepthStream );
		}

		// depth to pcl
		for( int y = 0; y < nHeight - 1; y++ )
			for( int x = 0; x < nWidth - 1; x++ )
			{
				int idx = x + y * nWidth;

				int self  = s_buffer[ idx ];

				s_pcl[ idx ] = float3
				(
					(x - 255.5f) / 369.21f * self,
					(211.5f - y) / 369.21f * self,
					self
				);

			}

		// Bilateral filter
		int   const kernel = 2;
		float const dfmm   = 0.4f;
		for( int y = 0; y < nHeight; y++ )
			for( int x = 0; x < nWidth; x++ )
			{
				int idx = x + y * nWidth;

				int xmin = maximum( 0, x - kernel );
				int xmax = minimum( nWidth - 1, x + kernel );

				int ymin = maximum( 0, y - kernel );
				int ymax = minimum( nHeight - 1, y + kernel );

				float intensity  = 0.0f;
				float sumWeights = 0.0f;

				float3 self = s_pcl[ idx ];
				if( self.z() > 0.0f )
				for( int j = ymin; j <= ymax; j++ )
					for( int i = xmin; i <= xmax; i++ )
					{
						int index = i + j * nWidth;

						float weight = 1.0f / fastexp( length( self - s_pcl[ index ] ) * dfmm );

						intensity += weight * s_buffer[ index ];
						sumWeights += weight * (s_buffer[ index ] != 0.0f);
					}

				sumWeights += sumWeights == 0.0f;
				intensity /= sumWeights;

				s_buffer2[ idx ] = (unsigned short) intensity;
			}

		// Visualize
#if 0 // depth map
		for( int i = 0, res = nWidth * nHeight; i < res; i++ )
		{
			RGBQUAD color;
			intensity_to_rgb( s_buffer2[ i ], nMinDepth, nMaxDepth, color.rgbRed, color.rgbGreen, color.rgbBlue );

			m_pDepthRGBX[ i ] = color;
		}
#else // normal map
		for( int y = 0; y < nHeight - 1; y++ )
			for( int x = 0; x < nWidth - 1; x++ )
			{
				int idx = x + y * nWidth;

				RGBQUAD color;
				* reinterpret_cast< int * >( & color ) = 0;

				m_pDepthRGBX[ idx ] = color;

				int self  = s_buffer2[ idx ];
				int right = s_buffer2[ idx + 1 ];
				int down  = s_buffer2[ idx + nWidth ];

				if( 0 == self || 0 == right || 0 == down )
					continue;

				float3 pself
				(
					(x - 255.5f) / 369.21f * self,
					(211.5f - y) / 369.21f * self,
					self
				);

				float3 pright
				(
					(x - 254.5f) / 369.21f * right,
					(211.5f - y) / 369.21f * right,
					right
				);

				float3 pdown
				(
					(x - 255.5f) / 369.21f * down,
					(210.5f - y) / 369.21f * down,
					down
				);

				float3 n = normalize( cross( pright - pself, pdown - pself ) );
				n.z() = -n.z();
				n = n * 0.5f + 0.5f;

				color.rgbRed   = n.x() * 255;
				color.rgbGreen = n.y() * 255;
				color.rgbBlue  = n.z() * 255;

				m_pDepthRGBX[ idx ] = color;
			}
#endif

		if (m_hWnd)
		{
			POINT cursorPosition;
			GetCursorPos( & cursorPosition );
			ScreenToClient( m_hWnd, & cursorPosition );

			int nx = cursorPosition.x / 768.0f * nWidth;
			int ny = cursorPosition.y / 688.0f * nHeight;

			nx = clamp( nx, 0, nWidth  - 1 );
			ny = clamp( ny, 0, nHeight - 1 );

			unsigned intensity = s_buffer2[ nx + ny * nWidth ];

			WCHAR szStatusMessage[64];
			StringCchPrintf(szStatusMessage, _countof(szStatusMessage), L"%dmm", intensity );

			if (SetStatusMessage(szStatusMessage, 100, false))
				m_nFramesSinceUpdate = 0;
		}

        // Draw the data with Direct2D
        m_pDrawDepth->Draw(reinterpret_cast<BYTE*>(m_pDepthRGBX), cDepthWidth * cDepthHeight * sizeof(RGBQUAD));

		// button pressed
        if (m_bSaveScreenshot)
        {
            if( m_bRecording )
			{
				fclose( m_pDepthStream );
				m_pDepthStream = nullptr;
			}
			else
			{
				m_pDepthStream = fopen( "I:/tmp/test.depth", "wb" );

				int v = 2;
				int w = nWidth;
				int h = nHeight;
				int T = 0;
				int n = 100;

				fwrite( "KPPL raw depth\n", 1, 15, m_pDepthStream );
				fwrite( & v, 4, 1, m_pDepthStream );
				fwrite( & w, 4, 1, m_pDepthStream );
				fwrite( & h, 4, 1, m_pDepthStream );
				fwrite( & T, 4, 1, m_pDepthStream );
				fwrite( & n, 4, 1, m_pDepthStream );
			}

            // toggle off so we don't save a screenshot again next frame
            m_bSaveScreenshot = false;
			m_bRecording = ! m_bRecording;
        }
    }
}

/// <summary>
/// Set the status bar message
/// </summary>
/// <param name="szMessage">message to display</param>
/// <param name="showTimeMsec">time in milliseconds to ignore future status messages</param>
/// <param name="bForce">force status update</param>
bool CDepthBasics::SetStatusMessage(_In_z_ WCHAR* szMessage, DWORD nShowTimeMsec, bool bForce)
{
    INT64 now = GetTickCount64();

    if (m_hWnd && (bForce || (m_nNextStatusTime <= now)))
    {
        SetDlgItemText(m_hWnd, IDC_STATUS, szMessage);
        m_nNextStatusTime = now + nShowTimeMsec;

        return true;
    }

    return false;
}

/// <summary>
/// Get the name of the file where screenshot will be stored.
/// </summary>
/// <param name="lpszFilePath">string buffer that will receive screenshot file name.</param>
/// <param name="nFilePathSize">number of characters in lpszFilePath string buffer.</param>
/// <returns>
/// S_OK on success, otherwise failure code.
/// </returns>
HRESULT CDepthBasics::GetScreenshotFileName(_Out_writes_z_(nFilePathSize) LPWSTR lpszFilePath, UINT nFilePathSize)
{
    WCHAR* pszKnownPath = NULL;
    HRESULT hr = SHGetKnownFolderPath(FOLDERID_Pictures, 0, NULL, &pszKnownPath);

    if (SUCCEEDED(hr))
    {
        // Get the time
        WCHAR szTimeString[MAX_PATH];
        GetTimeFormatEx(NULL, 0, NULL, L"hh'-'mm'-'ss", szTimeString, _countof(szTimeString));

        // File name will be KinectScreenshotDepth-HH-MM-SS.bmp
        StringCchPrintfW(lpszFilePath, nFilePathSize, L"%s\\KinectScreenshot-Depth-%s.bmp", pszKnownPath, szTimeString);
    }

    if (pszKnownPath)
    {
        CoTaskMemFree(pszKnownPath);
    }

    return hr;
}

/// <summary>
/// Save passed in image data to disk as a bitmap
/// </summary>
/// <param name="pBitmapBits">image data to save</param>
/// <param name="lWidth">width (in pixels) of input image data</param>
/// <param name="lHeight">height (in pixels) of input image data</param>
/// <param name="wBitsPerPixel">bits per pixel of image data</param>
/// <param name="lpszFilePath">full file path to output bitmap to</param>
/// <returns>indicates success or failure</returns>
HRESULT CDepthBasics::SaveBitmapToFile(BYTE* pBitmapBits, LONG lWidth, LONG lHeight, WORD wBitsPerPixel, LPCWSTR lpszFilePath)
{
    DWORD dwByteCount = lWidth * lHeight * (wBitsPerPixel / 8);

    BITMAPINFOHEADER bmpInfoHeader = {0};

    bmpInfoHeader.biSize        = sizeof(BITMAPINFOHEADER);  // Size of the header
    bmpInfoHeader.biBitCount    = wBitsPerPixel;             // Bit count
    bmpInfoHeader.biCompression = BI_RGB;                    // Standard RGB, no compression
    bmpInfoHeader.biWidth       = lWidth;                    // Width in pixels
    bmpInfoHeader.biHeight      = -lHeight;                  // Height in pixels, negative indicates it's stored right-side-up
    bmpInfoHeader.biPlanes      = 1;                         // Default
    bmpInfoHeader.biSizeImage   = dwByteCount;               // Image size in bytes

    BITMAPFILEHEADER bfh = {0};

    bfh.bfType    = 0x4D42;                                           // 'M''B', indicates bitmap
    bfh.bfOffBits = bmpInfoHeader.biSize + sizeof(BITMAPFILEHEADER);  // Offset to the start of pixel data
    bfh.bfSize    = bfh.bfOffBits + bmpInfoHeader.biSizeImage;        // Size of image + headers

    // Create the file on disk to write to
    HANDLE hFile = CreateFileW(lpszFilePath, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);

    // Return if error opening file
    if (NULL == hFile) 
    {
        return E_ACCESSDENIED;
    }

    DWORD dwBytesWritten = 0;
    
    // Write the bitmap file header
    if (!WriteFile(hFile, &bfh, sizeof(bfh), &dwBytesWritten, NULL))
    {
        CloseHandle(hFile);
        return E_FAIL;
    }
    
    // Write the bitmap info header
    if (!WriteFile(hFile, &bmpInfoHeader, sizeof(bmpInfoHeader), &dwBytesWritten, NULL))
    {
        CloseHandle(hFile);
        return E_FAIL;
    }
    
    // Write the RGB Data
    if (!WriteFile(hFile, pBitmapBits, bmpInfoHeader.biSizeImage, &dwBytesWritten, NULL))
    {
        CloseHandle(hFile);
        return E_FAIL;
    }    

    // Close the file
    CloseHandle(hFile);
    return S_OK;
}
