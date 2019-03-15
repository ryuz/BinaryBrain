#include <iostream>
#include <chrono>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bbcu/bbcu.h"
#include "bbcu/bbcu_util.h"



//////////////////////////////
// forward
//////////////////////////////

__global__ void kernal_fp32_Col2Im_Forward(
			const float*	x_buf,
            float*			y_buf,          
			int				hw_size,
            int				c_size,
            int				output_frame_size,
            int				output_frame_stride,
            int				input_frame_stride
		)
{
    int output_frame = blockDim.x * blockIdx.x + threadIdx.x;
    int xy           = blockDim.y * blockIdx.y + threadIdx.y;
    int c            = blockDim.z * blockIdx.z + threadIdx.z;

    if (output_frame < output_frame_size && xy < hw_size ) {
        int output_node = c * hw_size + xy;
        int input_frame = output_frame * hw_size + xy;
        int input_node  = c;

        y_buf[output_node * output_frame_stride + output_frame] = x_buf[input_node * input_frame_stride + input_frame];
    }
}


CUBB_DLL_EXPORT int cubb_fp32_Col2Im_Forward
		(
			float const     *dev_x_buf,
            float           *dev_y_buf,
			int			    w_size,
			int			    h_size,
			int			    c_size,
            int			    input_frame_stride,
            int			    output_frame_size,
			int			    output_frame_stride,
            cudaStream_t	streamId
		)
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());
    
    int     hw_size = h_size * w_size;
    
	dim3	block(32, 32, 1);
	dim3	grid((output_frame_size+31)/32, (hw_size+31)/32, c_size);
	
	kernal_fp32_Col2Im_Forward<<<grid, block, 0, streamId>>>(
			dev_x_buf,
            dev_y_buf,          
			hw_size,
            c_size,
            output_frame_size,
            output_frame_stride,
            input_frame_stride
		);
	BB_CUDA_CHECK_LAST_ERROR();

	return 0;
}


//////////////////////////////
// backward
//////////////////////////////

__global__ void kernal_fp32_Col2Im_Backward(
			const float*	dy_buf,
            float*			dx_buf,          
			int				hw_size,
            int				c_size,
            int				output_frame_size,
            int				output_frame_stride,
            int				input_frame_stride
		)
{
    int output_frame = blockDim.x * blockIdx.x + threadIdx.x;
    int xy           = blockDim.y * blockIdx.y + threadIdx.y;
    int c            = blockDim.z * blockIdx.z + threadIdx.z;

    if (output_frame < output_frame_size && xy < hw_size ) {
        int output_node = c * hw_size + xy;
        int input_frame = output_frame * hw_size + xy;
        int input_node  = c;

         dx_buf[input_node * input_frame_stride + input_frame] = dy_buf[output_node * output_frame_stride + output_frame];
    }
}

CUBB_DLL_EXPORT int cubb_fp32_Col2Im_Backward
		(
			float const     *dev_dy_buf,
            float           *dev_dx_buf,
			int			    w_size,
			int			    h_size,
			int			    c_size,
            int			    input_frame_stride,
            int			    output_frame_size,
			int			    output_frame_stride,
            cudaStream_t	streamId
		)
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());
    
    int     hw_size = h_size * w_size;
    
	dim3	block(32, 32, 1);
	dim3	grid((output_frame_size+31)/32, (hw_size+31)/32, c_size);
	
	kernal_fp32_Col2Im_Backward<<<grid, block, 0, streamId>>>(
			dev_dy_buf,
            dev_dx_buf,          
			hw_size,
            c_size,
            output_frame_size,
            output_frame_stride,
            input_frame_stride
		);
	BB_CUDA_CHECK_LAST_ERROR();

	return 0;
}


