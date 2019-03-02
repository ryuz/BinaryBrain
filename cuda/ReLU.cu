#include <iostream>
#include <chrono>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bbcu/bbcu.h"
#include "bbcu/bbcu_util.h"




//////////////////////////////
// forward
//////////////////////////////

__global__ void kernal_fp32_ReLU_Forward(
			const float*	x_buf,
			float*			y_buf,
			int				frame_stride
		)
{
	int frame = blockDim.x * blockIdx.x + threadIdx.x;
	int node  = blockDim.y * blockIdx.y + threadIdx.y;

    float x = x_buf[frame_stride*node + frame];
    if ( x <= 0 ) { x = 0; }
    y_buf[frame_stride*node + frame] = x;
}


CUBB_DLL_EXPORT int cubb_fp32_ReLU_Forward
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
	
	kernal_fp32_ReLU_Forward<<<grid, block, 0, streamId>>>(
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

__global__ void kernal_fp32_ReLU_Backward
        (
			const float*	x_buf,
			const float*    dy_buf,
		    float*	        dx_buf,
			int				frame_stride
		)
{
	int frame = blockDim.x * blockIdx.x + threadIdx.x;
	int node  = blockDim.y * blockIdx.y + threadIdx.y;
	
    float x  = x_buf[frame_stride*node + frame];
    float dy = dy_buf[frame_stride*node + frame];
    if ( x <= 0 ) { dy = 0; }
    dx_buf[frame_stride*node + frame] = dy;
}


CUBB_DLL_EXPORT int cubb_fp32_ReLU_Backward
		(
			const float*	dev_x_buf,
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
	
	kernal_fp32_ReLU_Backward<<<grid, block, 0, streamId>>>(
			dev_x_buf,
            dev_dy_buf,
            dev_dx_buf,
			frame_stride
		);
	BB_CUDA_CHECK_LAST_ERROR();

	return 0;
}

// end of file
