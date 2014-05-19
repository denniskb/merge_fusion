#include <cassert>



template< typename T, class Alloc >
svc::vector2d< T, Alloc >::vector2d( Alloc const & allocator ) :
	std::vector< T, Alloc >( allocator ),
	m_width( 0 )
{
}

template< typename T, class Alloc >
svc::vector2d< T, Alloc >::vector2d
(
	size_t width, size_t height,
	T const & val, 
	Alloc const & allocator 
) :
	std::vector< T, Alloc >( width * height, val, allocator ),
	m_width( width )
{
}



template< typename T, class Alloc >
size_t svc::vector2d< T, Alloc >::width() const
{
	return m_width;
}

template< typename T, class Alloc >
size_t svc::vector2d< T, Alloc >::height() const
{
	return width() ? size() / width() : 0;
}



template< typename T, class Alloc >
T & svc::vector2d< T, Alloc >::operator()( size_t x, size_t y )
{
	assert( x < width() );
	assert( y < height() );

	return (*this)[ x + y * width() ];
}

template< typename T, class Alloc >
T const & svc::vector2d< T, Alloc >::operator()( size_t x, size_t y ) const
{
	assert( x < width() );
	assert( y < height() );

	return (*this)[ x + y * width() ];
}



template< typename T, class Alloc >
void svc::vector2d< T, Alloc >::resize( size_t width, size_t height, T const & val )
{
	m_width = width;

	std::vector< T, Alloc >::resize( width * height, val );
}