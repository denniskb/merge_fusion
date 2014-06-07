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



#include "vector2d.inl"