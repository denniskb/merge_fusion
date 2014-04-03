#include "Cache.h"

#include <cassert>

#include <flink/util.h>

#include "Voxel.h"



svc::Cache::Cache() :
	m_sliceRes( 0 )
{
}



void svc::Cache::Reset( int sliceRes )
{
	assert( sliceRes > 0 );

	m_cachedSlices.resize( sliceRes * sliceRes * 2 );
	if( sliceRes > m_sliceRes )
		std::memset( m_cachedSlices.begin(), 0, m_cachedSlices.size() * sizeof( unsigned ) );

	m_sliceRes = sliceRes;

	m_slice0 = m_cachedSlices.begin();
	m_slice1 = m_cachedSlices.begin() + SliceSize();

	m_slice0range = std::make_pair( 0, 0 );
	m_slice1range = std::make_pair( 0, 0 );
}

bool svc::Cache::NextSlice
(
	unsigned const * keys,
	Voxel const * values,
	int numEntries
)
{
	for( int i = m_slice0range.first; i < m_slice0range.second; i++ )
	{
		unsigned x, y, z;
		flink::unpackInts( keys[ i ], x, y, z );
		m_slice0[ x + m_sliceRes * y ] = 0;
	}

	if( m_slice0range.second == numEntries )
		return false;

	std::swap( m_slice0, m_slice1 );
	m_slice0range = m_slice1range;

	if( m_slice0range.second == numEntries )
		return true;
	
	m_slice1range.first = m_slice1range.second;
	m_slice1range.second = (int) ( std::lower_bound
	(
		keys, 
		keys + numEntries, 
		flink::packInts( 0, 0, flink::unpackZ( keys[ m_slice1range.first ] ) + 1 )
	) - keys );

	for( int i = m_slice1range.first; i < m_slice1range.second; i++ )
	{
		unsigned x, y, z;
		flink::unpackInts( keys[ i ], x, y, z );
		m_slice1[ x + m_sliceRes * y ] = values[ i ];
	}

	if( m_slice0range.second == 0 )
		return NextSlice( keys, values, numEntries );

	return true;
}



std::pair< svc::Voxel const *, svc::Voxel const * > svc::Cache::CachedSlices() const
{
	return std::make_pair( m_slice0, m_slice1 );
}

std::pair< int, int > svc::Cache::CachedRange() const
{
	return m_slice0range;
}



int svc::Cache::SliceRes() const
{
	return m_sliceRes;
}

int svc::Cache::SliceSize() const
{
	return SliceRes() * SliceRes();
}