#pragma once

#include <kifi/cuda/device_allocator.h>



namespace kifi {
namespace cuda {

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

}} // namespace



#pragma region Implementation

#include <algorithm>



namespace kifi {
namespace cuda {

template< typename T, class Alloc >
buffer< T, Alloc >::buffer( size_t capacity, Alloc const & alloc ) :
	m_capacity( capacity ),
	m_alloc( alloc )
{
	m_data = m_alloc.allocate( capacity );
}

template< typename T, class Alloc >
buffer< T, Alloc >::buffer( buffer && move )
{
	swap( move );
}

template< typename T, class Alloc >
buffer< T, Alloc >::~buffer()
{
	m_alloc.deallocate( m_data, m_capacity );
}



template< typename T, class Alloc >
size_t buffer< T, Alloc >::capacity() const
{
	return m_capacity;
}

template< typename T, class Alloc >
T * buffer< T, Alloc >::data()
{
	return m_data;
}

template< typename T, class Alloc >
T const * buffer< T, Alloc >::data() const
{
	return m_data;
}



template< typename T, class Alloc >
void buffer< T, Alloc >::swap( buffer & rhs )
{
	using std::swap;

	swap( m_alloc, rhs.m_alloc );
	swap( m_data, rhs.m_data );
	swap( m_capacity, rhs.m_capacity );
}

template< typename T, class Alloc >
buffer< T, Alloc > & buffer< T, Alloc >::operator=( buffer && rhs )
{
	swap( rhs );
	return * this;
}

}} // namespace

#pragma endregion