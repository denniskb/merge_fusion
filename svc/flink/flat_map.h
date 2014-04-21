#pragma once

#include <algorithm>
#include <cassert>
#include <vector>

#include "algorithm.h"



namespace flink {

template< typename K, typename T >
class flat_map
{
public:
	typedef typename std::vector< K >::const_iterator const_key_iterator;
	typedef typename std::vector< T >::iterator value_iterator;
	typedef typename std::vector< T >::const_iterator const_value_iterator;

	inline const_key_iterator keys_cbegin() const { return m_keys.cbegin(); }
	inline const_key_iterator keys_cend()   const { return m_keys.cend();   }

	inline value_iterator values_begin() { return m_values.begin(); }
	inline value_iterator values_end()   { return m_values.end();   }
	inline const_value_iterator values_begin()  const { return m_values.begin();  }
	inline const_value_iterator values_end()    const { return m_values.end();    }
	inline const_value_iterator values_cbegin() const { return m_values.cbegin(); }
	inline const_value_iterator values_cend()   const { return m_values.cend();   }



	template< class BidirectionalIterator >
	inline void merge_unique
	(
		BidirectionalIterator keys_first2, BidirectionalIterator keys_last2,
		T const & value
	)
	{
		size_t intersection = intersection_size
		(
			keys_cbegin(), keys_cend(),
			keys_first2, keys_last2
		);
		
		size_t oldSize = size();
		resize( oldSize + std::distance( keys_first2, keys_last2 ) - intersection );

		merge_unique_backward
		(
			keys_cbegin(), keys_cbegin() + oldSize,
			values_cbegin() + oldSize,

			keys_first2, keys_last2,
			value,

			keys_end(), values_end()
		);
	}



	inline size_t size() const
	{
		return m_keys.size();
	}

	inline void clear()
	{
		m_keys.clear();
		m_values.clear();
	}

private:
	typedef typename std::vector< K >::iterator key_iterator;

	std::vector< K > m_keys;
	std::vector< T > m_values;

	inline void resize( size_t newSize )
	{
		m_keys.resize( newSize );
		m_values.resize( newSize );
	}

	inline key_iterator keys_begin() { return m_keys.begin(); }
	inline key_iterator keys_end()   { return m_keys.end();   }
	inline const_key_iterator keys_begin() const { return m_keys.begin(); }
	inline const_key_iterator keys_end()   const { return m_keys.end();   }
};

}