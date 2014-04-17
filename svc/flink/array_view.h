#pragma once

#include <cassert>



namespace flink {

template< typename T, unsigned width, unsigned height = 1, unsigned depth = 1 >
class array_view
{
public:
	inline explicit array_view( T * data ) :
		m_data( data )
	{
	}

	

	inline T & operator[]( unsigned i )
	{
		assert( i < width * height * depth );

		return m_data[ i ];
	}



	inline T & operator()( unsigned x )
	{
		assert( x < width );

		return m_data[ x ];
	}

	inline T & operator()( unsigned x, unsigned y )
	{
		assert( x < width );
		assert( y < height );

		return m_data[ x + y * width ];
	}

	inline T & operator()( unsigned x, unsigned y, unsigned z )
	{
		assert( x < width );
		assert( y < height );
		assert( z < depth );

		return m_data[ x + y * width + z * width * height ];
	}

private:
	T * m_data;
};

}