#pragma once

#include <cassert>

#include "vector.h"



namespace flink {

template< typename K, typename T >
class flat_map
{
public:
	flat_map(){}

	/*
	@precond keys.size == values.size
	@precond keys is sorted ascending
	@precond keys contains no duplicates
	*/
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