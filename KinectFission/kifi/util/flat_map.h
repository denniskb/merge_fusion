#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

#include <kifi/util/iterator.h>



namespace kifi {
namespace util {

template< typename Key, typename T >
class flat_map
{
public:
	typedef typename std::vector< Key >::iterator key_iterator;
	typedef typename std::vector< Key >::const_iterator const_key_iterator;
	typedef typename std::vector< T >::iterator value_iterator;
	typedef typename std::vector< T >::const_iterator const_value_iterator;

private:
	key_iterator keys_begin(); 
	key_iterator keys_end();  
public:
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

	void clear();

	template< class InputIterator1, class InputIterator2 >
	void insert
	(
		InputIterator1 keys_first, InputIterator1 keys_last,
		InputIterator2 values_first
	);

	std::size_t size() const;

private:
	std::vector< Key > m_keys;
	std::vector< T > m_values;

	std::vector< Key > m_tmpKeys;
	std::vector< T > m_tmpValues;
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
void flat_map< Key, T >::clear()
{
	m_keys.clear();
	m_values.clear();
}

template< typename Key, typename T >
template< class InputIterator1, class InputIterator2 >
void flat_map< Key, T >::insert
(
	InputIterator1 keys_first, InputIterator1 keys_last,
	InputIterator2 values_first
)
{	
	std::size_t nEntries = std::distance( keys_first, keys_last );
	std::size_t conservativeNewSize = size() + nEntries;

	m_tmpKeys.  resize( conservativeNewSize );
	m_tmpValues.resize( conservativeNewSize );

	auto end = set_union
	(
		m_keys.cbegin(), m_keys.cend(),
		keys_first, keys_last,

		values_cbegin(),
		values_first,

		m_tmpKeys.begin(),
		m_tmpValues.begin()
	);

	size_t actualNewSize = std::distance( m_tmpKeys.begin(), end );

	m_tmpKeys.  resize( actualNewSize );
	m_tmpValues.resize( actualNewSize );

	m_keys.  swap( m_tmpKeys   );
	m_values.swap( m_tmpValues );
}

template< typename Key, typename T >
size_t flat_map< Key, T >::size() const
{
	return m_keys.size();
}

}} // namespace

#pragma endregion