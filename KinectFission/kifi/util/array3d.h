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



#include "array3d.inl"