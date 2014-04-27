#pragma once

#include <vector>



namespace svc {

template< typename Key, typename T >
class flat_map
{
public:
	typedef typename std::vector< Key >::const_iterator const_key_iterator;
	typedef typename std::vector< T >::iterator value_iterator;
	typedef typename std::vector< T >::const_iterator const_value_iterator;

	inline const_key_iterator keys_begin() const;
	inline const_key_iterator keys_end()   const;
	inline const_key_iterator keys_cbegin() const;
	inline const_key_iterator keys_cend()   const;

	inline value_iterator values_begin();
	inline value_iterator values_end();
	inline const_value_iterator values_begin()  const;
	inline const_value_iterator values_end()    const;
	inline const_value_iterator values_cbegin() const;
	inline const_value_iterator values_cend()   const;

	template< class BidirectionalIterator >
	inline void merge_unique
	(
		BidirectionalIterator keys_first2, BidirectionalIterator keys_last2,
		T const & value
	);

	inline size_t size() const;

	inline void clear();

private:
	typedef typename std::vector< Key >::iterator key_iterator;

	std::vector< Key > m_keys;
	std::vector< T > m_values;

	inline key_iterator keys_begin(); 
	inline key_iterator keys_end();   

	inline void resize( size_t newSize );
};

}



#include "flat_map.inl"