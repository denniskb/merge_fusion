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

	inline float & operator[]( int i );
	inline float   operator[]( int i ) const;
};

struct float4
{
	float x, y, z, w;

	inline float4();
	inline float4( float s );
	inline float4( float x, float y, float z, float w );

	inline float & operator[]( int i );
	inline float   operator[]( int i ) const;

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
inline void     eigen             ( float4x4 const & m, float4 & outEigenValues, float4x4 & outEigenVectors );
inline float4x4 invert_transform  ( float4x4 const & tR );
inline float4x4 perspective_fov_rh( float fovYradians, float aspectWbyH, float nearZdistance, float farZdistance );
inline float4x4 perspective_fl_pp_rh
(
	float focalLengthXPixels   , float focalLengthYPixels, 
	float principalPointXPixels, float principalPointYPixels,	
	float nearZdistance        , float farZdistance 
);
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
#include <functional>



namespace kifi {
namespace util {

#pragma region float4/float4x4

float3::float3() {}
float3::float3( float s ) : x( s ), y( s ), z( s ) {}
float3::float3( float x, float y, float z ) : x( x ), y( y ), z( z ) {}

float & float3::operator[]( int i )
{
	assert( i >= 0 && i < 3 );

	return reinterpret_cast< float * >( this )[ i ];
}

float float3::operator[]( int i ) const
{
	assert( i >= 0 && i < 3 );

	return reinterpret_cast< float const * >( this )[ i ];
}



float4::float4() {}
float4::float4( float s ) : x( s ), y( s ), z( s ), w( s ) {}
float4::float4( float x, float y, float z, float w ) : x( x ), y( y ), z( z ), w( w ) {}
float3 float4::xyz() const { return float3( x, y, z ); }

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
	assert( iRow >= 0 && iRow < 4 );
	assert( iCol >= 0 && iCol < 4 );

	return m_data[ iCol + 4 * iRow ];
}

float float4x4::operator()( int iRow, int iCol ) const
{
	assert( iRow >= 0 && iRow < 4 );
	assert( iCol >= 0 && iCol < 4 );

	return m_data[ iCol + 4 * iRow ];
}

float4 float4x4::row( int i ) const
{
	assert( i >= 0 && i < 4 );

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
	assert( i >= 0 && i < 4 );

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



float4 operator*( float4x4 m, float4 v )
{
	return float4
	(
		dot( m.row( 0 ), v ),
		dot( m.row( 1 ), v ),
		dot( m.row( 2 ), v ),
		dot( m.row( 3 ), v )
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

float4x4 invert_transform( float4x4 const & tR )
{
	float4x4 R( tR );
	R( 0, 3 ) = 0.0f;
	R( 1, 3 ) = 0.0f;
	R( 2, 3 ) = 0.0f;

	float4x4 tInv = identity;
	tInv( 0, 3 ) = -tR( 0, 3 );
	tInv( 1, 3 ) = -tR( 1, 3 );
	tInv( 2, 3 ) = -tR( 2, 3 );

	transpose( R );
	return R * tInv;
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
	result( 2, 3 ) = -Q * nearZdistance;
	result( 3, 2 ) = -1.0f;

	return result;
}

float4x4 perspective_fl_pp_rh
(
	float focalLengthXPixels,
	float focalLengthYPixels,

	float principalPointXPixels,
	float principalPointYPixels,

	float nearZdistance,
	float farZdistance
)
{
	float Q = farZdistance / (farZdistance - nearZdistance);
	
	float4x4 result;
	std::memset( & result, 0, 64 );

	result( 0, 0 ) = focalLengthXPixels / principalPointXPixels;
	result( 1, 1 ) = focalLengthYPixels / principalPointYPixels;
	result( 2, 2 ) = -Q;
	result( 2, 3 ) = -Q * nearZdistance;
	result( 3, 2 ) = -1.0f;

	return result;
}

void transpose( float4x4 & m )
{
	for( int row = 0; row < 4; row++ )
		for( int col = row + 1; col < 4; col++ )
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

	result.col0 = set( m.col( 0 ) );
	result.col1 = set( m.col( 1 ) );
	result.col2 = set( m.col( 2 ) );
	result.col3 = set( m.col( 3 ) );

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