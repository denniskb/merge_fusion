#pragma once

#include <cstddef>
#include <memory>



namespace svcu {

template< typename T >
class device_allocator : public std::allocator< T >
{
public:
	pointer allocate( size_t n, const_pointer hint = 0 );
	void deallocate( pointer p, size_t n );
};

}



#include "device_allocator.inl"