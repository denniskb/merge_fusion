#pragma once

#include <vector>



namespace kifi {
namespace util {

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

}} // namespace



#pragma region Implementation

#include <kifi/util/algorithm.h>



namespace kifi {
namespace util {

template< typename Key, typename T >
typename flat_map< Key, T >::key_iterator 
flat_map< Key, T >::keys_begin()
{
	return m_keys.begin();
}

template< typename Key, typename T >
typename flat_map< Key, T >::key_iterator 
flat_map< Key, T >::keys_end()
{
	return m_keys.end();
}

template< typename Key, typename T >
typename flat_map< Key, T >::const_key_iterator 
flat_map< Key, T >::keys_begin() const
{
	return m_keys.begin();
}

template< typename Key, typename T >
typename flat_map< Key, T >::const_key_iterator 
flat_map< Key, T >::keys_end() const
{
	return m_keys.end();
}

template< typename Key, typename T >
typename flat_map< Key, T >::const_key_iterator 
flat_map< Key, T >::keys_cbegin() const
{
	return m_keys.cbegin();
}

template< typename Key, typename T >
typename flat_map< Key, T >::const_key_iterator
flat_map< Key, T >::keys_cend() const
{
	return m_keys.cend();   
}

template< typename Key, typename T >
typename flat_map< Key, T >::value_iterator 
flat_map< Key, T >::values_begin()
{
	return m_values.begin();
}

template< typename Key, typename T >
typename flat_map< Key, T >::value_iterator 
flat_map< Key, T >::values_end() 
{
	return m_values.end();   
}

template< typename Key, typename T >
typename flat_map< Key, T >::const_value_iterator 
flat_map< Key, T >::values_begin() const 
{
	return m_values.begin();  
}

template< typename Key, typename T >
typename flat_map< Key, T >::const_value_iterator 
flat_map< Key, T >::values_end() const 
{
	return m_values.end();    
}

template< typename Key, typename T >
typename flat_map< Key, T >::const_value_iterator 
flat_map< Key, T >::values_cbegin() const 
{
	return m_values.cbegin(); 
}

template< typename Key, typename T >
typename flat_map< Key, T >::const_value_iterator 
flat_map< Key, T >::values_cend() const
{
	return m_values.cend();   
}



template< typename Key, typename T >
template< class BidirectionalIterator >
void flat_map< Key, T >::merge_unique
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

	set_union_backward
	(
		keys_cbegin(), keys_cbegin() + oldSize,
		values_cbegin() + oldSize,

		keys_first2, keys_last2,
		value,

		keys_end(), values_end()
	);
}

template< typename Key, typename T >
size_t flat_map< Key, T >::size() const
{
	return m_keys.size();
}

template< typename Key, typename T >
void flat_map< Key, T >::clear()
{
	m_keys.clear();
	m_values.clear();
}

template< typename Key, typename T >
void flat_map< Key, T >::resize( size_t newSize )
{
	m_keys.resize( newSize );
	m_values.resize( newSize );
}

}} // namespace

#pragma endregion