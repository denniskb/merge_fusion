#pragma once

#include <array>

#include <xmmintrin.h> // SSE1



namespace kifi {
namespace util {

#pragma region Types

struct int2
{
	int x, y;

	inline int2();
	inline int2( int i );
	inline int2( int x, int y );
};

struct float2
{
	float x, y;

	inline float2();
	inline float2( float s );
	inline float2( float x, float y );
};

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
	inline float4( float3 xyz, float w );

	inline float & operator[]( int i );
	inline float   operator[]( int i ) const;

	inline float4 operator-() const;
};



struct float4x4
{
	float4 col0, col1, col2, col3;

	inline float4x4();
	explicit inline float4x4( float s );
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
	inline float const & operator()( int iRow, int iCol ) const;
};

float4x4 const identity
(
	1.0f, 0.0f, 0.0f, 0.0f,
	0.0f, 1.0f, 0.0f, 0.0f,
	0.0f, 0.0f, 1.0f, 0.0f,
	0.0f, 0.0f, 0.0f, 1.0f
);



typedef __m128 vector;
struct matrix { vector col0, col1, col2, col3; };

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

inline float4   operator*( float4x4 m, float4 v   );
inline float4x4 operator*( float4x4 m, float4x4 n );

inline vector operator* ( matrix m, vector v );

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

// Eigenvectors are stored in the columns of 'outEigenVectors'. 'm' and 'outEigenVectors' are allowed to be identical.
inline void eigen           ( float4x4 const & m, float4 & outEigenValues, float4x4 & outEigenVectors );
inline void invert_transform( float4x4 & tR );
inline void transpose       ( float4x4 & m );

inline int      all       ( vector v );
inline int      any       ( vector v );
template< int index >	  
inline vector   broadcast ( vector v );
inline vector   dot       ( vector u, vector v );
inline vector   homogenize( vector v );
inline vector   load      ( float4 src );
inline matrix   load      ( float4x4 src );
inline vector   load      ( float const * src );
inline vector   loadu     ( float const * src );
inline vector   loadss    ( float src );
inline int      none      ( vector v );
inline vector   set       ( float s );
inline vector   set       ( float x, float y, float z, float w );
template< int a0, int a1, int b0, int b1 >
inline vector   shuffle   ( vector a, vector b );
inline float4   store     ( vector src );
inline void     store     ( float * dst, vector src );
inline void     storeu    ( float * dst, vector src );
inline float    storess   ( vector src );
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
#include <functional>



namespace kifi {
namespace util {

#pragma region Types

int2::int2() {}
int2::int2( int i ) : x( i ), y( i ) {}
int2::int2( int x, int y ) : x( x ), y( y ) {}

float2::float2() {}
float2::float2( float s ) : x( s ), y( s ) {}
float2::float2( float x, float y ) : x( x ), y( y ) {}

float3::float3() {}
float3::float3( float s ) : x( s ), y( s ), z( s ) {}
float3::float3( float x, float y, float z ) : x( x ), y( y ), z( z ) {}

float4::float4() {}
float4::float4( float s ) : x( s ), y( s ), z( s ), w( s ) {}
float4::float4( float x, float y, float z, float w ) : x( x ), y( y ), z( z ), w( w ) {}
float4::float4( float3 xyz, float w ) : x( xyz.x ), y( xyz.y ), z( xyz.z ), w( w ) {}
float4 float4::operator-() const { return float4( -x, -y, -z, -w ); }



float & float4::operator[]( int i )
{
	assert( i >= 0 && i < 4 );

	return reinterpret_cast< float * >( this )[ i ];
}

float float4::operator[]( int i ) const
{
	assert( i >= 0 && i < 4 );

	return reinterpret_cast< float const * >( this )[ i ];
}



float4x4::float4x4() {}

float4x4::float4x4( float s ) :
	col0( s ),
	col1( s ),
	col2( s ),
	col3( s )
{
}

float4x4::float4x4
(
	float m00, float m01, float m02, float m03,
	float m10, float m11, float m12, float m13,
	float m20, float m21, float m22, float m23,
	float m30, float m31, float m32, float m33
) :
	col0( m00, m10, m20, m30 ),
	col1( m01, m11, m21, m31 ),
	col2( m02, m12, m22, m32 ),
	col3( m03, m13, m23, m33 )
{
}

float4x4::float4x4( float const * src )
{
	std::memcpy( this, src, 64 );
	transpose( * this );
}

float & float4x4::operator()( int iRow, int iCol )
{
	assert( iRow >= 0 && iRow < 4 );
	assert( iCol >= 0 && iCol < 4 );

	return reinterpret_cast< float * >( this )[ iRow + 4 * iCol ];
}

float const & float4x4::operator()( int iRow, int iCol ) const
{
	assert( iRow >= 0 && iRow < 4 );
	assert( iCol >= 0 && iCol < 4 );

	return reinterpret_cast< float const * >( this )[ iRow + 4 * iCol ];
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



float4 operator*( float4x4 m, float4 v )
{
	return ( m.col0 * v.x + m.col1 * v.y ) + ( m.col2 * v.z + m.col3 * v.w );
}

float4x4 operator*( float4x4 m, float4x4 n )
{
	float4x4 result;

	result.col0 = m * n.col0;
	result.col1 = m * n.col1;
	result.col2 = m * n.col2;
	result.col3 = m * n.col3;

	return result;
}



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



void eigen( float4x4 const & m, float4 & outEigenValues, float4x4 & outEigenVectors )
{
	float4 tmp;
	outEigenVectors = m;

	std::function< void ( float4x4 &, float4 &, float4 & ) > tred2;
	std::function< void ( float4 &, float4 &, float4x4 & ) > tqli;

	tred2( outEigenVectors, outEigenValues, tmp );
	tqli ( outEigenValues, tmp, outEigenVectors );

	// TODO: Replace copy-righted code.
#pragma region tred2/tqli implementation

	static int const MAX_ITERS = 30;
	auto SIGN = [] ( float a, float b ) { return b < 0.0f ? -std::abs( a ) : std::abs( a ); };

	tred2 = [] ( float4x4 & a, float4 & d, float4 & e )
	{
		int l, k, j, i;
        float scale, hh, h, g, f;

        for (i = 3; i >= 1; i--)
        {
            l = i - 1;
            h = scale = 0.0f;
            if (l > 0)
            {
                for (k = 0; k <= l; k++)
                    scale += std::abs(a(i,k));
                if (scale == 0.0f)
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
                    f = 0.0f;
                    for (j = 0; j <= l; j++)
                    {
                        /* Next statement can be omitted if eigenvectors not wanted */
                        a(j,i) = a(i,j) / h;
                        g = 0.0f;
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
        d[0] = 0.0f;
        e[0] = 0.0f;
        /* Contents of this loop can be omitted if eigenvectors not 
                wanted except for statement d[i]=a(i,i); */
        for (i = 0; i < 4; i++)
        {
            l = i - 1;
            if (d[i] != 0.0f)
            {
                for (j = 0; j <= l; j++)
                {
                    g = 0.0f;
                    for (k = 0; k <= l; k++)
                        g += a(i,k) * a(k,j);
                    for (k = 0; k <= l; k++)
                        a(k,j) -= g * a(k,i);
                }
            }
            d[i] = a(i,i);
            a(i,i) = 1.0f;
            for (j = 0; j <= l; j++) a(j,i) = a(i,j) = 0.0f;
        }
	};

	tqli = [ & SIGN ] ( float4 & d, float4 & e, float4x4 & z )
	{
		int m,l,iter,i,k;  
        float s,r,p,g,f,dd,c,b;  
      
        for (i=1;i<4;i++) e[i-1]=e[i];
        e[3] = 0.0f;
        for (l = 0; l < 4; l++)
        {  
            iter=0;  
            do {
                for (m = l; m < 3; m++)
                {
                    dd=std::abs(d[m])+std::abs(d[m+1]);  
                    if (std::abs(e[m])+dd == dd) break;  
                }  
                if (m != l) {  
                    if (iter++ == MAX_ITERS) return; /* Too many iterations in TQLI */  
                    g=(d[l+1]-d[l])/(2.0f*e[l]);  
                    r=std::sqrt((g*g)+1.0f);  
                    g=d[m]-d[l]+e[l]/(g+SIGN(r,g));  
                    s=c=1.0f;  
                    p=0.0f;  
                    for (i=m-1;i>=l;i--) {  
                        f=s*e[i];  
                        b=c*e[i];  
                        if (std::abs(f) >= std::abs(g)) {  
                            c=g/f;  
                            r=std::sqrt((c*c)+1.0f);  
                            e[i+1]=f*r;  
                            c *= (s=1.0f/r);  
                        } else {  
                            s=f/g;  
                            r=std::sqrt((s*s)+1.0f);  
                            e[i+1]=g*r;  
                            s *= (c=1.0f/r);  
                        }  
                        g=d[i+1]-p;  
                        r=(d[i]-g)*s+2.0f*c*b;  
                        p=s*r;  
                        d[i+1]=g+p;  
                        g=c*r-b;  
                        /* Next loop can be omitted if eigenvectors not wanted */
                        for (k = 0; k < 4; k++)
                        {  
                            f=z(k,i+1);  
                            z(k,i+1)=s*z(k,i)+c*f;  
                            z(k,i)=c*z(k,i)-s*f;  
                        }  
                    }  
                    d[l]=d[l]-p;  
                    e[l]=g;  
                    e[m]=0.0f;  
                }  
            } while (m != l);  
        }
	};

#pragma endregion
}

void invert_transform( float4x4 & tR )
{
	float4 tInv = -tR.col3; tInv.w = 1.0f;
	tR.col3 = float4( 0.0f, 0.0f, 0.0f, 1.0f );
	
	std::swap( tR.col0.y, tR.col1.x );
	std::swap( tR.col0.z, tR.col2.x );
	std::swap( tR.col1.z, tR.col2.y );

	tR.col3 = tR * tInv;
}

void transpose( float4x4 & m )
{
	std::swap( m.col0.y, m.col1.x );
	std::swap( m.col0.z, m.col2.x );
	std::swap( m.col0.w, m.col3.x );
			   
	std::swap( m.col1.z, m.col2.y );
	std::swap( m.col1.w, m.col3.y );
			   
	std::swap( m.col2.w, m.col3.z );
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

vector homogenize( vector v )
{
	return v / broadcast< 3 >( v );
}

vector load( float4 src )
{
	return loadu( reinterpret_cast< float * >( & src ) );
}

matrix load( float4x4 src )
{
	matrix result;

	result.col0 = load( src.col0 );
	result.col1 = load( src.col1 );
	result.col2 = load( src.col2 );
	result.col3 = load( src.col3 );

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