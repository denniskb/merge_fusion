#pragma once

#include "device_allocator.h"



namespace svcu {

template< typename T, class Alloc = device_allocator< T > >
class buffer
{
public:
	buffer( size_t capacity, Alloc const & alloc = Alloc() );
	buffer( buffer && move );
	~buffer();

	size_t capacity() const;
	T * data();
	T const * data() const;

	void swap( buffer & rhs );
	buffer & operator=( buffer && rhs );

private:
	Alloc m_alloc;
	T * m_data;
	size_t m_capacity;

	buffer( buffer const & );
	buffer & operator=( buffer );
};

}



#include "buffer.inl"