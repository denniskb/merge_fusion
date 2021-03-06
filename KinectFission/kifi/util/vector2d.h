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
		std::size_t width, std::size_t height,
		T const & val = T(),
		Alloc const & allocator = Alloc() 
	);

	std::size_t width() const;
	std::size_t height() const;
	
	T & operator()( std::size_t x, std::size_t y );
	T const & operator()( std::size_t x, std::size_t y ) const;

	void resize( std::size_t width, std::size_t height, T const & val = T() );

private:
	std::size_t m_width;
	std::size_t m_height;
};

}}



#pragma region Implementation

#include <cassert>



namespace kifi {
namespace util {

template< typename T, class Alloc >
vector2d< T, Alloc >::vector2d( Alloc const & allocator ) :
	std::vector< T, Alloc >( allocator ),
	m_width( 0 ),
	m_height( 0 )
{
}

template< typename T, class Alloc >
vector2d< T, Alloc >::vector2d
(
	std::size_t width, std::size_t height,
	T const & val, 
	Alloc const & allocator 
) :
	std::vector< T, Alloc >( width * height, val, allocator ),
	m_width( width ),
	m_height( height )
{
}



template< typename T, class Alloc >
std::size_t vector2d< T, Alloc >::width() const
{
	return m_width;
}

template< typename T, class Alloc >
std::size_t vector2d< T, Alloc >::height() const
{
	return m_height;
}



template< typename T, class Alloc >
T & vector2d< T, Alloc >::operator()( std::size_t x, std::size_t y )
{
	assert( x < width() );
	assert( y < height() );

	return (*this)[ y * width() + x ];
}

template< typename T, class Alloc >
T const & vector2d< T, Alloc >::operator()( std::size_t x, std::size_t y ) const
{
	assert( x < width() );
	assert( y < height() );

	return (*this)[ y * width() + x ];
}



template< typename T, class Alloc >
void vector2d< T, Alloc >::resize( std::size_t width, std::size_t height, T const & val )
{
	m_width = width;
	m_height = height;

	std::vector< T, Alloc >::resize( width * height, val );
}

}} // namespace

#pragma endregion