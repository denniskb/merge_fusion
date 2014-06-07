#pragma once

#include <array>
#include <cstddef>



namespace kifi {
namespace util {

template< typename T, size_t width, size_t height, size_t depth >
class array3d : public std::array< T, width * height * depth >
{
public:
	T & operator()( size_t x, size_t y, size_t z );
	T const & operator()( size_t x, size_t y, size_t z ) const;
};

}} // namespace




#pragma region Implementation

#include <cassert>



namespace kifi {
namespace util {

template< typename T, size_t width, size_t height, size_t depth >
T & array3d< T, width, height, depth >::operator()
( 
	size_t x, 
	size_t y, 
	size_t z 
)
{
	assert( x < width );
	assert( y < height );
	assert( z < depth );

	return (*this)[ x + y * width + z * width * height ];
}

template< typename T, size_t width, size_t height, size_t depth >
T const & array3d< T, width, height, depth >::operator()
( 
	size_t x, 
	size_t y, 
	size_t z 
) const
{
	assert( x < width );
	assert( y < height );
	assert( z < depth );

	return (*this)[ x + y * width + z * width * height ];
}

}} // namespace

#pragma endregion