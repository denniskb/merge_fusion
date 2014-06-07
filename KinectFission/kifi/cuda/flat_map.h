#pragma once

#include <algorithm>
#include <cstddef>

#include <thrust/device_vector.h>
#include <thrust/distance.h>
#include <thrust/set_operations.h>



namespace kifi {
namespace cuda {

template< typename Key, typename T >
class flat_map
{
public:
	inline size_t size() const
	{
		return m_keys.size();
	}

	template< class InputIterator1, class InputIterator2 >
	inline void merge_unique
	(
		InputIterator1 keys_first, InputIterator1 keys_last,
		InputIterator2 values_first
	)
	{
		int tmpSize = std::max( 1, m_tmpKeys.size() );
		while( tmpSize < size() + thrust::distance( keys_first, keys_last ) )
			tmpSize *= 2;

		m_tmpKeys.resize( tmpSize );
		m_tmpValues.resize( tmpSize );

		auto newSize = thrust::set_union_by_key
		(
			m_keys.cbegin(), m_keys.cend(),
			keys_first, keys_last,
			
			m_values.cbegin(),
			values_first,

			m_tmpKeys.begin(),
			m_tmpValues.begin()
		);

		m_tmpKeys.resize( thrust::distance( m_tmpKeys.cbegin(), newSize.first ) );
		m_tmpValues.resize( thrust::distance( m_tmpValues.cbegin(), newSize.second ) );

		m_keys.swap( m_tmpKeys );
		m_values.swap( m_tmpValues );
	}

private:
	thrust::device_vector< Key > m_keys;
	thrust::device_vector< T > m_values;

	thrust::device_vector< Key > m_tmpKeys;
	thrust::device_vector< T > m_tmpValues;
};

}} // namespace