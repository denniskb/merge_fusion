#pragma once

#include <array>
#include <cstddef>



namespace svc {

template< typename T, size_t width, size_t height, size_t depth >
class array3d : public std::array< T, width * height * depth >
{
public:
	inline T & operator()( size_t x, size_t y, size_t z );
	inline T const & operator()( size_t x, size_t y, size_t z ) const;
};

}



#include "array3d.inl"