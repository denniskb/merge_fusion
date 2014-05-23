#pragma once



inline __device__ unsigned lanemask_le()
{
	unsigned result;

	asm( "mov.u32 %0, %lanemask_le;" : "=r"( result ) );

	return result;
}

inline __device__ unsigned lanemask_lt()
{
	unsigned result;

	asm( "mov.u32 %0, %lanemask_lt;" : "=r"( result ) );

	return result;
}

inline __device__ unsigned laneid()
{
	unsigned result;

	asm( "mov.u32 %0, %laneid;" : "=r"( result ) );

	return result;
}

inline __device__ unsigned warpid()
{
	return threadIdx.x / WARP_SZ;
}

inline __device__ unsigned __shfl( unsigned var, int srcLane )
{
	asm( "shfl.idx.b32 %0, %0, %1, 0x1f;" : "+r"( var ) : "r"( srcLane ) );

	return var;
}

inline __device__ unsigned __shfl_xor( unsigned var, int laneMask )
{
	asm( "shfl.bfly.b32 %0, %0, %1, 0x1f;" : "+r"( var ) : "r"( laneMask ) );

	return var;
}