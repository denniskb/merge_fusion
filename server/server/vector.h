/*
Never-shrinking vector with minimal effort.
Only suited for PODs!!!
*/

#pragma once

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>



namespace kppl {

template< typename T >
class vector
{
public:
	inline vector() : 
		m_data( nullptr ), 
		m_capacity( 0 ),
		m_size( 0 )
	{
	}

	inline vector( int initialSize ) : 
		m_data( nullptr ), 
		m_capacity( 0 ),
		m_size( 0 )
	{
		resize( initialSize );
	}
	
	inline vector( vector const & copy ) :
		m_data( (T*) std::malloc( copy.m_capacity * sizeof( T ) ) ),
		m_capacity( copy.m_capacity ),
		m_size( copy.m_size )
	{
		std::memcpy( m_data, copy.m_data, copy.m_size * sizeof( T ) );
	}

	inline vector( vector&& copy ) :
		m_data( nullptr ),
		m_capacity( 0 ),
		m_size( 0 )
	{
		swap( * this, copy );
	}

	inline vector & operator=( vector rhs )
	{
		swap( * this, rhs );
		return * this;
	}

	inline ~vector()
	{
		if( m_data != nullptr )
			std::free( m_data );
	}



	inline int size() const
	{
		return m_size;
	}

	inline T operator[]( int index ) const
	{
		assert( index >= 0 && index < m_size );

		return m_data[ index ];
	}

	inline T & operator[]( int index )
	{
		assert( index >= 0 && index < m_size );

		return m_data[ index ];
	}



	inline T * begin()
	{
		return m_data;
	}

	inline T * end()
	{
		return m_data + m_size;
	}

	inline T const * cbegin()
	{
		return m_data;
	}

	inline T const * cend()
	{
		return m_data + m_size;
	}



	inline void push_back( T element )
	{
		int newSize = m_size + 1;
		if( newSize > m_capacity )
			reserve( 2 * newSize );

		m_data[ m_size ] = element;
		m_size = newSize;
	}

	inline void reserve( int newCapacity )
	{
		if( newCapacity > m_capacity )
		{
			m_data = (T*) std::realloc( m_data, newCapacity * sizeof( T ) );
			m_capacity = newCapacity;
		}
	}

	inline void resize( int newSize )
	{
		reserve( newSize );
		m_size = newSize;
	}

	inline void clear()
	{
		m_size = 0;
	}

private:
	T * m_data;
	int m_capacity;
	int m_size;

friend void swap( vector & first, vector & second )
{
	using std::swap;

	swap( first.m_data, second.m_data );
	swap( first.m_capacity, second.m_capacity );
	swap( first.m_size, second.m_size );
}
};

}