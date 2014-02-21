#pragma once



namespace kppl {

struct KernelDepthFrame
{
	float const * data;
	int width;
	int height;

	inline KernelDepthFrame( float const * data, int width, int height ) :
		data( data ), width( width ), height( height )
	{
	}

#ifdef __CUDACC__

	__device__ float operator()( int x, int y ) const
	{
		return data[ x + y * width ];
	}

#endif
};

}