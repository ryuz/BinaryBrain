#include <iostream>
#include <chrono>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bbcu/bbcu.h"
#include "bbcu/bbcu_util.h"


#if 0


//////////////////////////////
// forward
//////////////////////////////

__global__ void kernal_fp32_BatchNormalization_Forward(
			const float*	x_buf,
			float*			y_buf,
			float*			tmp_buf,
			int				frame_stride
		)
{
	int frame = blockDim.x * blockIdx.x + threadIdx.x;
	int node  = blockDim.y * blockIdx.y + threadIdx.y;

	// ì«Ç›çûÇ›
	float acc = 0;
	const float* src_ptr = &src[size * y];
	while (x < size) {
		acc += src_ptr[x];
		x += x_step;
	}
	buf[threadIdx.x] = acc;

	__syncthreads();

	x = threadIdx.x;
	int comb = 1;
	while (comb < size) {
		int next = comb * 2;
		int mask = next - 1;
		if ((x & mask) == 0) {
			buf[x] += buf[x + comb];
		}
		comb = next;
		__syncthreads();
	}

	dst[y] = buf[0];
}


CUBB_DLL_EXPORT int cubb_fp32_BatchNormalization_Forward
		(
			const float*	dev_x_buf,
			float*			dev_y_buf,
			int				frame_size,
			int				frame_stride,
			int				node_size,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

	int		frame_block = frame_size;
	int		frame_grid  = 1;
    while (frame_block > 1024) {
        frame_block /= 2;
        frame_grid  *= 2;
    }

	dim3	grid(frame_grid, node_size);
	dim3	block(frame_block, 1);
	
	kernal_fp32_Binarize_Forward<<<grid, block, 0, streamId>>>(
			dev_x_buf,
            dev_y_buf,
			frame_stride
		);
	BB_CUDA_CHECK_LAST_ERROR();

	return 0;
}


//////////////////////////////
// backward
//////////////////////////////

__global__ void kernal_fp32_BatchNormalization_Backward
        (
			const float*	x_buf,
			const float*	y_buf,
			const float*    dy_buf,
		    float*	        dx_buf,
			int				frame_stride
		)
{
	int frame = blockDim.x * blockIdx.x + threadIdx.x;
	int node  = blockDim.y * blockIdx.y + threadIdx.y;
	

}


CUBB_DLL_EXPORT int cubb_fp32_BatchNormalization_Backward
		(
			const float*	dev_x_buf,
			const float*	dev_y_buf,
			const float*	dev_dy_buf,
			float*			dev_dx_buf,
			int				frame_size,
			int				frame_stride,
			int				node_size,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

	int		frame_block = frame_size;
	int		frame_grid  = 1;
    while (frame_block > 1024) {
        frame_block /= 2;
        frame_grid  *= 2;
    }

	dim3	grid(frame_grid, node_size);
	dim3	block(frame_block, 1);
	
	kernal_fp32_BatchNormalization_Backward<<<grid, block, 0, streamId>>>(
			dev_x_buf,
            dev_dy_buf,
            dev_dx_buf,
			frame_stride
		);
	BB_CUDA_CHECK_LAST_ERROR();

	return 0;
}

#endif