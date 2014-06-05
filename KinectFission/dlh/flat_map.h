#pragma once

#include <vector>



namespace dlh {

template< typename Key, typename T >
class flat_map
{
public:
	typedef typename std::vector< Key >::const_iterator const_key_iterator;
	typedef typename std::vector< T >::iterator value_iterator;
	typedef typename std::vector< T >::const_iterator const_value_iterator;

	const_key_iterator keys_begin() const;
	const_key_iterator keys_end()   const;
	const_key_iterator keys_cbegin() const;
	const_key_iterator keys_cend()   const;

	value_iterator values_begin();
	value_iterator values_end();
	const_value_iterator values_begin() const;
	const_value_iterator values_end()   const;
	const_value_iterator values_cbegin() const;
	const_value_iterator values_cend()   const;

	template< class BidirectionalIterator >
	void merge_unique
	(
		BidirectionalIterator keys_first2, BidirectionalIterator keys_last2,
		T const & value
	);

	size_t size() const;

	void clear();

private:
	typedef typename std::vector< Key >::iterator key_iterator;

	std::vector< Key > m_keys;
	std::vector< T > m_values;

	key_iterator keys_begin(); 
	key_iterator keys_end();   

	void resize( size_t newSize );
};

}



#include "flat_map.inl"