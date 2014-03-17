#pragma once

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <type_traits>



namespace svc {

template< typename T >
class vector
{
public:
	vector( int size = 0 ) :
		m_data( nullptr ),
		m_size( 0 ),
		m_capacity( 0 )
	{
		static_assert( std::is_pod< T >::value, "vector can only hold PODs." );

		resize( size );
	}

	~vector()
	{
		if( m_data != nullptr )
			std::free( m_data );
	}

	vector( vector const & copy ) :
		m_data( nullptr ),
		m_size( 0 ),
		m_capacity( 0 )
	{
		static_assert( std::is_pod< T >::value, "vector can only hold PODs." );

		copy_from( copy );
	}

	vector( vector && rhs )
	{
		static_assert( std::is_pod< T >::value, "vector can only hold PODs." );

		using std::swap;

		swap( * this, rhs );
	}

	vector& operator=( vector const & rhs )
	{
		copy_from( rhs );

		return * this;
	}

	friend void swap( vector & lhs, vector & rhs )
	{
		using std::swap;

		swap( lhs.m_data, rhs.m_data );
		swap( lhs.m_capacity, rhs.m_capacity );
		swap( lhs.m_size, rhs.m_size );
	}



	int size() const
	{
		return m_size;
	}

	T * begin()
	{
		return m_data;
	}
	T const * cbegin() const
	{
		return m_data;
	}

	T * end()
	{
		return m_data + size();
	}
	T const * cend() const
	{
		return m_data + size();
	}

	T & operator[]( int index )
	{
		return m_data[ index ];
	}
	T operator[]( int index ) const
	{
		return m_data[ index ];
	}



	void reserve( int capacity )
	{
		if( capacity > m_capacity )
		{
			m_data = (T*) std::realloc( m_data, capacity * sizeof( T ) );
			m_capacity = capacity;
		}
	}

	void resize( int size )
	{
		reserve( size );
		m_size = size;
	}

	void clear()
	{
		m_size = 0;
	}

	void push_back( T element )
	{
		int newSize = size() + 1;
		
		if( newSize > m_capacity )
			reserve( 2 * newSize );

		m_data[ size() ] = element;
		m_size = newSize;
	}

private:
	T * m_data;
	int m_size;
	int m_capacity;

	void copy_from( vector< T > const & rhs )
	{
		resize( rhs.size() );
		std::memcpy( m_data, rhs.m_data, rhs.size() * sizeof( T ) );
	}
};

}