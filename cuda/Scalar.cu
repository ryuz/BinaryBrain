#include <iostream>
#include <chrono>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bbcu/bbcu.h"
#include "bbcu/bbcu_util.h"



// -------------------------------------------------
//  add_ex
// -------------------------------------------------

__global__ void kernal_Scalar_add_ex
		(
			float*			dst,
			const float*	src0,
			const float*	src1,
			float			a,
			float			b,
			float			c,
			int				size
		)
{
    int	index = threadIdx.x;
	while ( index < size ) {
		dst[index] = a * src0[index] + b * src1[index] + c;
		index += blockDim.x;
	}
}

CUBB_DLL_EXPORT int bbcu_Scalar_add_ex(
			float*			dev_dst,
			const float*	dev_src0,
			const float*	dev_src1,
			float			a,
			float			b,
			float			c,
			int				size,
            cudaStream_t	streamId
		)
{
	kernal_Scalar_add_ex<<<1, 1024, 0, streamId>>>
        (
			dev_dst,
			dev_src0,
			dev_src1,
			a,
			b,
			c,
			size
		);
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}



// -------------------------------------------------
//  mul_ex
// -------------------------------------------------

__global__ void kernal_Scalar_mul_ex(
			float*			dst,
			const float*	src0,
			const float*	src1,
			float			a,
			float			b,
			int				size)
{
	int	index = threadIdx.x;
	while ( index < size ) {
		dst[index] = a * src0[index] * src1[index] + b;
		index += blockDim.x;
	}
}


CUBB_DLL_EXPORT int bbcu_Scalar_mul_ex
        (
			float*			dev_dst,
			const float*	dev_src0,
			const float*	dev_src1,
			float			a,
			float			b,
			int				size,
            cudaStream_t	streamId
		)
{
	kernal_Scalar_mul_ex<<<1, 1024, 0, streamId>>>
        (
			dev_dst,
			dev_src0,
			dev_src1,
			a,
			b,
			size
		);
    BB_CUDA_CHECK_LAST_ERROR();

    return 0;
}


// end of file
