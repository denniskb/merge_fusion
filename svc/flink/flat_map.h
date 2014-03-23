#pragma once

#include <cassert>

#include "vector.h"



namespace flink {

template< typename K, typename T >
class flat_map
{
public:
	flat_map(){}

	flat_map( vector< K > && keys, vector< T > && values ) :
		m_keys( std::move( keys ) ),
		m_values( std::move( values ) )
	{
		assert( keys.size() == values.size() );
	}



	inline int size() const
	{
		return m_keys.size();
	}



	inline vector< K > const & keys() const
	{
		return m_keys;
	}

	inline vector< T > const & values() const
	{
		return m_values;
	}



	inline void insert( flat_map< K, T > const & other )
	{
		int oldSize = size();
		resize( size() + other.size() );

		int i, j, k;
		i = oldSize - 1;
		j = other.size() - 1;
		k = size() - 1;

		while( i >= 0 && j >= 0 )
		{
			if( m_keys[ i ] > other.m_keys[ j ] )
			{
				m_keys[ k ] = std::move( m_keys[ i ] );
				m_values[ k ] = std::move( m_values[ i ] );
				i--;
			}
			else
			{
				m_keys[ k ] = other.m_keys[ j ];
				m_values[ k ] = other.m_values[ j ];
				j--;
			}
			k--;
		}

		while( i >= 0 )
		{
			m_keys[ k ] = std::move( m_keys[ i ] );
			m_values[ k ] = std::move( m_values[ i ] );
			i--;
			k--;
		}

		while( j >= 0 )
		{
			m_keys[ k ] = other.m_keys[ j ];
			m_values[ k ] = other.m_values[ j ];
			j--;
			k--;
		}
	}

	inline void remove( vector< K > const & keys )
	{
		if( 0 == keys.size() )
			return;

		int i = 0, k = 0;
		
		for( int j = 0; j < m_keys.size(); j++ )
		{
			while( k < keys.size() - 1 && keys[ k ] < m_keys[ j ] ){ k++; }

			if( m_keys[ j ] != keys[ k ] )
			{
				m_keys[ i ] = std::move( m_keys[ j ] );
				m_values[ i ] = std::move( m_values[ j ] );
				i++;
			}
		}

		resize( i );
	}



	inline void clear()
	{
		m_keys.clear();
		m_values.clear();
	}

private:
	vector< K > m_keys;
	vector< T > m_values;

	inline void resize( int newSize )
	{
		m_keys.resize( newSize );
		m_values.reserve( newSize );
	}
};

}