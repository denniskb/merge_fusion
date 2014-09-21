#pragma once

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
	return std::max( a, std::min( x, b ) );
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