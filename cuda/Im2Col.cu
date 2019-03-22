#include <iostream>
#include <chrono>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bbcu/bbcu.h"
#include "bbcu/bbcu_util.h"




//////////////////////////////
// forward
//////////////////////////////

__global__ void kernal_fp32_Im2Col_Forward(
			const float*	input_sig_buf,
			int				input_frame_stride,
			int				input_w_size,
			int				input_h_size,
            
			float*			output_sig_buf,
			int				output_frame_size,
			int				output_frame_stride,
			int				output_w_size,
			int				output_size
		)
{
	int filter_w_size = blockDim.y;
	int filter_h_size = blockDim.z;

	int output_frame = blockDim.x * blockIdx.x + threadIdx.x;
	int fx           = threadIdx.y;
	int fy           = threadIdx.z;
	int c            = blockIdx.y;
	
    if ( output_frame < output_frame_size ) {
	    int input_frame = output_frame / output_size;
	    int f           = output_frame % output_size;
	    int ix = f % output_w_size + fx;
	    int iy = f / output_w_size + fy;

	    int input_node  = (c * input_h_size  + iy) * input_w_size  + ix;
	    float sig = input_sig_buf[input_node * input_frame_stride + input_frame];

	    int output_node = (c * filter_h_size + fy) * filter_w_size + fx;	
        output_sig_buf[output_node * output_frame_stride + output_frame] = sig;
    }
}


BBCU_DLL_EXPORT int bbcu_fp32_Im2Col_Forward
		(
			const float*	input_sig_dev_buf,
			int				input_frame_size,
			int				input_frame_stride,
			int				input_w_size,
			int				input_h_size,
			int				input_c_size,

			float*			output_sig_dev_buf,
			int				output_frame_stride,

			int				filter_w_size,
			int				filter_h_size,

            cudaStream_t	streamId
		)
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

	int output_c_size = input_c_size;
	int output_w_size = input_w_size - filter_w_size + 1;
	int output_h_size = input_h_size - filter_h_size + 1;
	int output_size   = output_w_size * output_h_size;
	
	int output_frame_size = input_frame_size * output_size;
    
	int		frame_unit = 16;
	dim3	grid((output_frame_size + (frame_unit-1))/frame_unit, output_c_size);
	dim3	block(frame_unit, filter_w_size, filter_h_size);
	
	kernal_fp32_Im2Col_Forward<<<grid, block, 0, streamId>>>(
			input_sig_dev_buf,
            input_frame_stride,
			input_w_size,
			input_h_size,
            
			output_sig_dev_buf,
			output_frame_size,
			output_frame_stride,
			output_w_size,
			output_size
		);
	BB_CUDA_CHECK_LAST_ERROR();

	return 0;
}



//////////////////////////////
// backward
//////////////////////////////

__global__ void kernal_fp32_Im2Col_Backward(
			float*			input_grad_buf,
			int				input_frame_size,
			int				input_frame_stride,
			int				input_w_size,
			int				input_h_size,
            
			const float*	output_grad_buf,
			int				output_frame_size,
			int				output_frame_stride,
			int				output_w_size,
			int				output_h_size,

			int				filter_w_size,
			int				filter_h_size
		)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	int c = blockDim.z * blockIdx.z + threadIdx.z;
	
	float const *output_grad_ptr = &output_grad_buf[c * filter_h_size * filter_w_size * output_frame_stride];

	for ( int input_frame = 0; input_frame < input_frame_size; ++input_frame ) {
		float grad = 0;
		for (int fy = 0; fy < filter_h_size; ++fy) {
			int iy = y - fy;
			if ( iy >= 0 && iy < (input_h_size - filter_h_size + 1)) {
				for (int fx = 0; fx < filter_w_size; ++fx) {
					int ix = x - fx;
					if (ix >= 0 && ix < (input_w_size - filter_w_size + 1)) {
						int output_frame = (input_frame * output_h_size + iy) * output_w_size + ix;
						int output_node  = fy * filter_w_size + fx;
						grad += output_grad_ptr[output_node * output_frame_stride + output_frame];
					}
				}
			}
		}
//		float*	input_grad_ptr  = &input_grad_buf[c * input_h_size * input_w_size * input_frame_stride];
//		int input_node  = y * input_w_size + x;
//		input_grad_ptr[input_node * input_frame_stride + input_frame] = grad;
		input_grad_buf[((c * input_h_size + y) * input_w_size + x) * input_frame_stride + input_frame] = grad;
	}
}


BBCU_DLL_EXPORT int bbcu_fp32_Im2Col_Backward
		(
			float*			input_grad_dev_buf,
			int				input_frame_size,
			int				input_frame_stride,
			int				input_w_size,
			int				input_h_size,
			int				input_c_size,
			const float*	output_grad_dev_buf,
			int				output_frame_stride,			
            int				filter_w_size,
			int				filter_h_size,            
            cudaStream_t	streamId
		)
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

//	int output_c_size = input_c_size;
	int output_w_size = input_w_size - filter_w_size + 1;
	int output_h_size = input_h_size - filter_h_size + 1;
	int output_size   = output_w_size * output_h_size;
	
	int output_frame_size = input_frame_size * output_size;
	
	dim3	grid(input_w_size, input_h_size, 1);
	dim3	block(1, 1, input_c_size);
	
	kernal_fp32_Im2Col_Backward<<<grid, block, 0, streamId>>>(
			input_grad_dev_buf,
			input_frame_size,
			input_frame_stride,
			input_w_size,
			input_h_size,
			output_grad_dev_buf,
			output_frame_size,
			output_frame_stride,
			output_w_size,
			output_h_size,
			filter_w_size,
			filter_h_size
		);
	BB_CUDA_CHECK_LAST_ERROR();

	return 0;
}



