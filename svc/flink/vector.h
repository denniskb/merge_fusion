/*
Never-shrinking vector.
Only suited for POD-like types!
  - Must not have a user-defined constructor
  - resize merely allocates memory - it doesn't initialize it.
*/

#pragma once

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>



namespace flink {

template< typename T >
class vector
{
public:
	inline explicit vector( int size = 0 ) :
		m_data( nullptr ),
		m_size( 0 ),
		m_capacity( 0 )
	{
		assert( size >= 0 );

		resize( size );
	}

	inline vector( T const * first, T const * last ) :
		m_data( nullptr ),
		m_size( 0 ),
		m_capacity( 0 )
	{
		assert( first != nullptr );
		assert( first <= last );

		copy_from( first, last );
	}

	inline vector( vector const & copy ) :
		m_data( nullptr ),
		m_size( 0 ),
		m_capacity( 0 )
	{
		copy_from( copy.cbegin(), copy.cend() );
	}

	inline vector( vector && rhs ) :
		m_data( nullptr ),
		m_size( 0 ),
		m_capacity( 0 )
	{
		using std::swap;

		swap( * this, rhs );
	}

	inline ~vector()
	{
		if( m_data != nullptr )
			std::free( m_data );
	}

	inline vector & operator=( vector const & rhs )
	{
		copy_from( rhs.cbegin(), rhs.cend() );

		return * this;
	}

	inline vector & operator=( vector && rhs )
	{
		using std::swap;

		swap( * this, rhs );

		return * this;
	}

	inline friend void swap( vector & lhs, vector & rhs )
	{
		using std::swap;

		swap( lhs.m_data, rhs.m_data );
		swap( lhs.m_capacity, rhs.m_capacity );
		swap( lhs.m_size, rhs.m_size );
	}



	inline int size() const
	{
		return m_size;
	}

	inline T * begin()
	{
		return m_data;
	}
	inline T const * cbegin() const
	{
		return m_data;
	}

	inline T * end()
	{
		return m_data + size();
	}
	inline T const * cend() const
	{
		return m_data + size();
	}

	inline T & operator[]( int index )
	{
		assert( index >= 0 && index < size() );

		return m_data[ index ];
	}
	inline T const & operator[]( int index ) const
	{
		assert( index >= 0 && index < size() );

		return m_data[ index ];
	}



	inline void resize( int size )
	{
		assert( size >= 0 );

		if( nullptr == m_data )
			reserve( size );
		else if( size > m_capacity )
			reserve( 2 * size );

		m_size = size;
	}

	inline void clear()
	{
		m_size = 0;
	}

	inline void push_back( T element )
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

	inline void copy_from( T const * first, T const * last )
	{
		resize( (int) ( last - first ) );
		std::memcpy( m_data, first, size() * sizeof( T ) );
	}

	inline void reserve( int capacity )
	{
		assert( capacity >= 0 );

		if( capacity > m_capacity )
		{
			m_data = (T*) std::realloc( m_data, capacity * sizeof( T ) );
			m_capacity = capacity;
		}
	}
};

}