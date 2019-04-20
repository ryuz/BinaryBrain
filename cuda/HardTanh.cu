#include <iostream>
#include <chrono>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bbcu/bbcu.h"
#include "bbcu/bbcu_util.h"




//////////////////////////////
// forward
//////////////////////////////

__global__ void kernal_fp32_HardTanh_Forward(
			float const *x_buf,
			float       *y_buf,
			int         frame_size,
			int         frame_stride
		)
{
 	int node    = blockIdx.x;
	int id      = threadIdx.x;
	int id_step = blockDim.x;
    
    for ( int frame = id; frame < frame_size; frame += id_step ) {
        float x = x_buf[frame_stride*node + frame];
        if (x < 0.0) { x = 0.0; }
        if (x > 1.0) { x = 1.0; }
        y_buf[frame_stride*node + frame] = x;
    }
}


BBCU_DLL_EXPORT int bbcu_fp32_HardTanh_Forward
		(
			float const *	dev_x_buf,
			float*			dev_y_buf,
			int				node_size,
			int				frame_size,
			int				frame_stride,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

	int		unit_x = 512;
	
	dim3	grid(node_size);
	dim3	block(unit_x);
	
	kernal_fp32_HardTanh_Forward<<<grid, block, 0, streamId>>>(
			dev_x_buf,
            dev_y_buf,
            frame_size,
			frame_stride
		);
	BB_CUDA_CHECK_LAST_ERROR();

	return 0;
}


//////////////////////////////
// backward
//////////////////////////////

__global__ void kernal_fp32_HardTanh_Backward
        (
			const float*	x_buf,
			const float*    dy_buf,
		    float*	        dx_buf,
			int				frame_size,
			int				frame_stride
		)
{
  	int node    = blockIdx.x;
	int id      = threadIdx.x;
	int id_step = blockDim.x;
    
    for ( int frame = id; frame < frame_size; frame += id_step ) {
        float x  = x_buf[frame_stride*node + frame];
        float dy = dy_buf[frame_stride*node + frame];
        if ( x <= -1.0f && x >= 1.0f) { dy = 0.0f; }
        dx_buf[frame_stride*node + frame] = dy;
    }
}


BBCU_DLL_EXPORT int bbcu_fp32_HardTanh_Backward
		(
			float const     *dev_x_buf,
			float const     *dev_dy_buf,
			float           *dev_dx_buf,
			int				node_size,
			int				frame_size,
			int				frame_stride,
            cudaStream_t    streamId
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

	int		unit_x = 512;
	
	dim3	grid(node_size);
	dim3	block(unit_x);

	kernal_fp32_HardTanh_Backward<<<grid, block, 0, streamId>>>(
			dev_x_buf,
            dev_dy_buf,
            dev_dx_buf,
            frame_size,
			frame_stride
		);
	BB_CUDA_CHECK_LAST_ERROR();

	return 0;
}

