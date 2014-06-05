#include <algorithm>



dlh::vec dlh::load( float4 const & v )
{
	return DirectX::XMLoadFloat4A( & v );
}

dlh::mat dlh::load( float4x4 const & m )
{
	return DirectX::XMLoadFloat4x4A( & m );
}

dlh::float4 dlh::store( vec v )
{
	float4 result;
	DirectX::XMStoreFloat4A( & result, v );
	return result;
}

dlh::float4x4 dlh::store( mat m )
{
	float4x4 result;
	DirectX::XMStoreFloat4x4A( & result, m );
	return result;
}



dlh::float4 dlh::make_float4( float s )
{
	return DirectX::XMFLOAT4A( s, s, s, s );
}

dlh::vec dlh::set( float x, float y, float z, float w )
{
	return DirectX::XMVectorSet( x, y, z, w );
}



template< typename T >
T dlh::clamp( T x, T a, T b )
{
	return std::max( a, std::min( x, b ) );
}

float dlh::dot( float4 const & a, float4 const & b )
{
	return
		a.x * b.x +
		a.y * b.y +
		a.z * b.z +
		a.w * b.w;
}

dlh::vec dlh::homogenize( vec v )
{
	return DirectX::XMVectorDivide( v, DirectX::XMVectorPermute< 3, 3, 3, 3 >( v, v ) );
}

float dlh::lerp( float a, float b, float weightA, float weightB )
{
	return a * weightA + b * weightB / ( weightA + weightB );
}

bool dlh::powerOf2( int x )
{
	return x > 0 && ! ( x & ( x - 1 ) );
}



unsigned dlh::packX( unsigned x )
{
	return x;
}

unsigned dlh::packY( unsigned y )
{
	return y << 10;
}

unsigned dlh::packZ( unsigned z )
{
	return z << 20;
}

unsigned dlh::packInts( unsigned x, unsigned y, unsigned z )
{
	return packX( x ) | packY( y ) | packZ( z );
}



unsigned dlh::unpackX( unsigned packedInt )
{
	return packedInt & 0x3ff;
}

unsigned dlh::unpackY( unsigned packedInt )
{
	return ( packedInt >> 10 ) & 0x3ff;
}

unsigned dlh::unpackZ( unsigned packedInt )
{
	return packedInt >> 20;
}

void dlh::unpackInts( unsigned packedInt, unsigned & outX, unsigned & outY, unsigned & outZ )
{
	outX = unpackX( packedInt );
	outY = unpackY( packedInt );
	outZ = unpackZ( packedInt );
}



dlh::float4 operator+( dlh::float4 const & a, dlh::float4 const & b )
{
	return dlh::float4
	(
		a.x + b.x,
		a.y + b.y,
		a.z + b.z,
		a.w + b.w
	);
}

dlh::float4 operator-( dlh::float4 const & a, dlh::float4 const & b )
{
	return dlh::float4
	(
		a.x - b.x,
		a.y - b.y,
		a.z - b.z,
		a.w - b.w
	);
}

dlh::float4 operator*( dlh::float4 const & a, dlh::float4 const & b )
{
	return dlh::float4
	(
		a.x * b.x,
		a.y * b.y,
		a.z * b.z,
		a.w * b.w
	);
}

dlh::float4 operator/( dlh::float4 const & a, dlh::float4 const & b )
{
	return dlh::float4
	(
		a.x / b.x,
		a.y / b.y,
		a.z / b.z,
		0.0f
	);
}



dlh::vec operator+( dlh::vec a, dlh::vec b )
{
	return DirectX::XMVectorAdd( a, b );
}

dlh::vec operator-( dlh::vec a, dlh::vec b )
{
	return DirectX::XMVectorSubtract( a, b );
}

dlh::vec operator*( dlh::vec a, dlh::vec b )
{
	return DirectX::XMVectorMultiply( a, b );
}

dlh::vec operator/( dlh::vec a, dlh::vec b )
{
	return DirectX::XMVectorDivide( a, b );
}



dlh::vec operator*( dlh::vec v, dlh::mat m )
{
	return DirectX::XMVector4Transform( v, m );
}