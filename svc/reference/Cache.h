#pragma once

#include <flink/vector.h>



namespace svc {

class Cache
{
public:
	Cache();

	void Reset( int sliceRes );
	bool NextSlice
	(
		unsigned const * keys,
		unsigned const * values,
		int numEntries
	);

	std::pair< unsigned const *, unsigned const * > CachedSlices() const;
	std::pair< int, int > CachedRange() const; // [ startIndex, endIndex )

	int SliceRes() const;
	int SliceSize() const;

private:
	int m_sliceRes;
	flink::vector< unsigned > m_cachedSlices;

	unsigned * m_slice0;
	unsigned * m_slice1;

	std::pair< int, int > m_slice0range;
	std::pair< int, int > m_slice1range;
};

}