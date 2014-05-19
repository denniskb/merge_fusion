#include <cassert>



template< typename T, size_t width, size_t height, size_t depth >
T & svc::array3d< T, width, height, depth >::operator()
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
T const & svc::array3d< T, width, height, depth >::operator()
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