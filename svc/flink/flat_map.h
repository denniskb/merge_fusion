#pragma once

#include <cassert>

#include "algorithm.h"
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
	flat_map( vector< K > keys, vector< T > values ) :
		m_keys( std::move( keys ) ),
		m_values( std::move( values ) )
	{
		assert( keys.size() == values.size() );
	}



	inline int size() const
	{
		return m_keys.size();
	}



	inline K const * keys_first() const { return m_keys.cbegin(); }
	inline K const * keys_last()  const { return m_keys.cend(); }

	inline T * values_first() { return m_values.begin(); }
	inline T * values_last()  { return m_values.end(); }
	inline T const * values_first() const { return m_values.cbegin(); }
	inline T const * values_last()  const { return m_values.cend(); }



	inline void merge_unique( K const * keys_first2, K const * keys_last2, T value )
	{
		if( 0 == keys_last2 - keys_first2 )
			return;

		if( 0 == size() )
		{
			m_keys = vector< K >( keys_first2, keys_last2 );
			m_values.resize( m_keys.size() );
			std::fill( m_values.begin(), m_values.end(), value );
			
			return;
		}

		int oldSize = size();
		int intersection = intersection_size( m_keys.cbegin(), m_keys.cend(), keys_first2, keys_last2 );
		resize( size() + (int)( keys_last2 - keys_first2 ) - intersection );

		K const * const merge_first = std::lower_bound( keys_first(), keys_first() + oldSize, * keys_first2 );
		K const * const merge_last  = std::upper_bound( keys_first(), keys_first() + oldSize, * ( keys_last2 - 1 ) );

		//std::copy_backward( merge_last, keys_first() + oldSize, m_keys.end() );
		//std::copy_backward
		//(
		//	values_first() + ( merge_last - keys_first() ),
		//	values_first() + oldSize,
		//	values_last()
		//);

		T const * values_last1 = values_first() + ( merge_last - keys_first() );

		int suffixLength = (int)( keys_first() + oldSize - merge_last );
		K * keys_result_last   = m_keys.end()  - suffixLength;
		T * values_result_last = values_last() - suffixLength;

		merge_unique_backward
		(
			merge_first, merge_last,
			values_last1,

			keys_first2, keys_last2,
			value,

			keys_result_last,
			values_result_last
		);
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
		m_values.resize( newSize );
	}
};

}