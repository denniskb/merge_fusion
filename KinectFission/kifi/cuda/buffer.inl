#include <algorithm>



template< typename T, class Alloc >
svcu::buffer< T, Alloc >::buffer( size_t capacity, Alloc const & alloc ) :
	m_capacity( capacity ),
	m_alloc( alloc )
{
	m_data = m_alloc.allocate( capacity );
}

template< typename T, class Alloc >
svcu::buffer< T, Alloc >::buffer( buffer && move )
{
	swap( move );
}

template< typename T, class Alloc >
svcu::buffer< T, Alloc >::~buffer()
{
	m_alloc.deallocate( m_data, m_capacity );
}



template< typename T, class Alloc >
size_t svcu::buffer< T, Alloc >::capacity() const
{
	return m_capacity;
}

template< typename T, class Alloc >
T * svcu::buffer< T, Alloc >::data()
{
	return m_data;
}

template< typename T, class Alloc >
T const * svcu::buffer< T, Alloc >::data() const
{
	return m_data;
}



template< typename T, class Alloc >
void svcu::buffer< T, Alloc >::swap( buffer & rhs )
{
	using std::swap;

	swap( m_alloc, rhs.m_alloc );
	swap( m_data, rhs.m_data );
	swap( m_capacity, rhs.m_capacity );
}

template< typename T, class Alloc >
svcu::buffer< T, Alloc > & svcu::buffer< T, Alloc >::operator=( buffer && rhs )
{
	swap( rhs );
	return * this;
}