#pragma once

#include <cstddef>
#include <vector>

#include <driver_types.h>

#include "buffer.h"
#include "kernel_vector.h"
#include "page_locked_allocator.h"



namespace svcu {

template< typename T >
class vector
{
public:
	vector( size_t initialCapacity = 0 );

	size_t capacity() const;
	void clear( cudaStream_t stream = 0 );

	T * data();
	T const * data() const;

	unsigned * size();
	unsigned const * size() const;

	svcu::kernel_vector< T > kernel_vector();
	svcu::kernel_vector< T > const kernel_vector() const;

	void reserve( size_t newCapacity );
	void resize( size_t newSize, cudaStream_t stream = 0 );

private:
	buffer< T > m_data;
	buffer< unsigned > m_size;

	buffer< unsigned, page_locked_allocator< unsigned > > m_tmpSize;

	vector( vector const & );
	vector & operator=( vector );
};

template< typename T >
void copy( vector< T > & dst, vector< T > const & src, cudaStream_t stream = 0 );

template< typename T >
void copy( std::vector< T > & dst, vector< T > const & src, cudaStream_t stream = 0 );

template< typename T >
void copy( vector< T > & dst, std::vector< T > const & src, cudaStream_t stream = 0 );

}



#include "vector.inl"