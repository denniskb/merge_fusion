#pragma once



inline __device__ unsigned laneid()
{
	unsigned result;

	asm( "{ mov.u32 %0, %laneid; }" : "=r"( result ) );

	return result;
}

inline __device__ unsigned warpid()
{
	return threadIdx.x / WARP_SZ;
}

inline __device__ unsigned __shfl( unsigned var, int const srcLane )
{
	asm
	(
		"{\n\t"
		"shfl.idx.b32 %0, %0, %1, 0x1f;\n\t"
		"}"
		:
		"+r"( var ) : "r"( srcLane )
	);

	return var;
}

inline __device__ unsigned __shfl_xor( unsigned var, int const laneMask )
{
	asm
	(
		"{\n\t"
		"shfl.bfly.b32 %0, %0, %1, 0x1f;\n\t"
		"}"
		:
		"+r"( var ) : "r"( laneMask )
	);

	return var;
}