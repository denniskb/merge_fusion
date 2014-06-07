#pragma once

#include <cstddef>
#include <memory>
#include <vector>



namespace kifi {
namespace util {

template< typename T, class Alloc = std::allocator< T > >
class vector2d : public std::vector< T, Alloc >
{
public:
	vector2d( Alloc const & allocator = Alloc() );

	vector2d
	( 
		size_t width, size_t height,
		T const & val = T(),
		Alloc const & allocator = Alloc() 
	);

	size_t width() const;
	size_t height() const;
	
	T & operator()( size_t x, size_t y );
	T const & operator()( size_t x, size_t y ) const;

	void resize( size_t width, size_t height, T const & val = T() );

private:
	size_t m_width;
};

}}



#pragma region Implementation

#include <cassert>



namespace kifi {
namespace util {

template< typename T, class Alloc >
vector2d< T, Alloc >::vector2d( Alloc const & allocator ) :
	std::vector< T, Alloc >( allocator ),
	m_width( 0 )
{
}

template< typename T, class Alloc >
vector2d< T, Alloc >::vector2d
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
size_t vector2d< T, Alloc >::width() const
{
	return m_width;
}

template< typename T, class Alloc >
size_t vector2d< T, Alloc >::height() const
{
	return width() ? size() / width() : 0;
}



template< typename T, class Alloc >
T & vector2d< T, Alloc >::operator()( size_t x, size_t y )
{
	assert( x < width() );
	assert( y < height() );

	return (*this)[ x + y * width() ];
}

template< typename T, class Alloc >
T const & vector2d< T, Alloc >::operator()( size_t x, size_t y ) const
{
	assert( x < width() );
	assert( y < height() );

	return (*this)[ x + y * width() ];
}



template< typename T, class Alloc >
void vector2d< T, Alloc >::resize( size_t width, size_t height, T const & val )
{
	m_width = width;

	std::vector< T, Alloc >::resize( width * height, val );
}

}} // namespace

#pragma endregion