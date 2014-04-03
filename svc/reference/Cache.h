#pragma once

#include <flink/vector.h>



namespace svc {

class Voxel;

class Cache
{
public:
	Cache();

	void Reset( int sliceRes );
	bool NextSlice
	(
		unsigned const * keys,
		Voxel const * values,
		int numEntries
	);

	std::pair< Voxel const *, Voxel const * > CachedSlices() const;
	std::pair< int, int > CachedRange() const; // [ startIndex, endIndex )

	int SliceRes() const;
	int SliceSize() const;

private:
	int m_sliceRes;
	flink::vector< Voxel > m_cachedSlices;

	Voxel * m_slice0;
	Voxel * m_slice1;

	std::pair< int, int > m_slice0range;
	std::pair< int, int > m_slice1range;
};

}