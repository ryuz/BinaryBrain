#include <iostream>
#include <chrono>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bbcu/bbcu.h"
#include "bbcu/bbcu_util.h"




//////////////////////////////
// forward
//////////////////////////////

__global__ void kernal_fp32_Binarize_Forward(
			float const *x_buf,
			float       *y_buf,
			int         frame_size,
			int         frame_stride
		)
{
 	int node       = blockIdx.x;
	int frame_base = threadIdx.x;
	int frame_step = blockDim.x;
    
    for ( int frame = frame_base; frame < frame_size; frame += frame_step ) {
        float x = x_buf[frame_stride*node + frame];
        x = (x > 0) ? 1 : 0;
        y_buf[frame_stride*node + frame] = x;
    }
}


BBCU_DLL_EXPORT int bbcu_fp32_Binarize_Forward
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
	
	kernal_fp32_Binarize_Forward<<<grid, block, 0, streamId>>>(
			dev_x_buf,
            dev_y_buf,
            frame_size,
			frame_stride
		);
	BB_CUDA_CHECK_LAST_ERROR();

	return 0;
}


