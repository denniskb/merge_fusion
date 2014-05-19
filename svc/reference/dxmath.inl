#include <algorithm>



svc::vec svc::load( float4 const & v )
{
	return DirectX::XMLoadFloat4A( & v );
}

svc::mat svc::load( float4x4 const & m )
{
	return DirectX::XMLoadFloat4x4A( & m );
}

svc::float4 svc::store( vec v )
{
	float4 result;
	DirectX::XMStoreFloat4A( & result, v );
	return result;
}

svc::float4x4 svc::store( mat m )
{
	float4x4 result;
	DirectX::XMStoreFloat4x4A( & result, m );
	return result;
}



svc::float4 svc::make_float4( float s )
{
	return DirectX::XMFLOAT4A( s, s, s, s );
}

svc::vec svc::set( float x, float y, float z, float w )
{
	return DirectX::XMVectorSet( x, y, z, w );
}



template< typename T >
T svc::clamp( T x, T a, T b )
{
	return std::max( a, std::min( x, b ) );
}

float svc::dot( float4 const & a, float4 const & b )
{
	return
		a.x * b.x +
		a.y * b.y +
		a.z * b.z +
		a.w * b.w;
}

svc::vec svc::homogenize( vec v )
{
	return DirectX::XMVectorDivide( v, DirectX::XMVectorPermute< 3, 3, 3, 3 >( v, v ) );
}

float svc::lerp( float a, float b, float weightA, float weightB )
{
	return a * weightA + b * weightB / ( weightA + weightB );
}

bool svc::powerOf2( int x )
{
	return x > 0 && ! ( x & ( x - 1 ) );
}



unsigned svc::packX( unsigned x )
{
	return x;
}

unsigned svc::packY( unsigned y )
{
	return y << 10;
}

unsigned svc::packZ( unsigned z )
{
	return z << 20;
}

unsigned svc::packInts( unsigned x, unsigned y, unsigned z )
{
	return packX( x ) | packY( y ) | packZ( z );
}



unsigned svc::unpackX( unsigned packedInt )
{
	return packedInt & 0x3ff;
}

unsigned svc::unpackY( unsigned packedInt )
{
	return ( packedInt >> 10 ) & 0x3ff;
}

unsigned svc::unpackZ( unsigned packedInt )
{
	return packedInt >> 20;
}

void svc::unpackInts( unsigned packedInt, unsigned & outX, unsigned & outY, unsigned & outZ )
{
	outX = unpackX( packedInt );
	outY = unpackY( packedInt );
	outZ = unpackZ( packedInt );
}



svc::float4 operator+( svc::float4 const & a, svc::float4 const & b )
{
	return svc::float4
	(
		a.x + b.x,
		a.y + b.y,
		a.z + b.z,
		a.w + b.w
	);
}

svc::float4 operator-( svc::float4 const & a, svc::float4 const & b )
{
	return svc::float4
	(
		a.x - b.x,
		a.y - b.y,
		a.z - b.z,
		a.w - b.w
	);
}

svc::float4 operator*( svc::float4 const & a, svc::float4 const & b )
{
	return svc::float4
	(
		a.x * b.x,
		a.y * b.y,
		a.z * b.z,
		a.w * b.w
	);
}

svc::float4 operator/( svc::float4 const & a, svc::float4 const & b )
{
	return svc::float4
	(
		a.x / b.x,
		a.y / b.y,
		a.z / b.z,
		0.0f
	);
}



svc::vec operator+( svc::vec a, svc::vec b )
{
	return DirectX::XMVectorAdd( a, b );
}

svc::vec operator-( svc::vec a, svc::vec b )
{
	return DirectX::XMVectorSubtract( a, b );
}

svc::vec operator*( svc::vec a, svc::vec b )
{
	return DirectX::XMVectorMultiply( a, b );
}

svc::vec operator/( svc::vec a, svc::vec b )
{
	return DirectX::XMVectorDivide( a, b );
}



svc::vec operator*( svc::vec v, svc::mat m )
{
	return DirectX::XMVector4Transform( v, m );
}