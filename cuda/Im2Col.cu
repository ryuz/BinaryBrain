#include <iostream>
#include <chrono>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bbcu/bbcu.h"
#include "bbcu/bbcu_util.h"


__global__ void kernal_Im2Col_Forward(
			const float*	in_sig,
			float*			out_sig,
			int				input_frame_size,
			int				input_w_size,
			int				input_h_size,
			int				output_frame_size,
			int				output_w_size,
			int				output_size
		)
{
//	int c_size        = gridDim.y;
	int filter_w_size = blockDim.y;
	int filter_h_size = blockDim.z;

	int output_frame = blockDim.x * blockIdx.x + threadIdx.x;
	int fx           = threadIdx.y;
	int fy           = threadIdx.z;
	int c            = blockIdx.y;
	
	int input_frame = output_frame / output_size;
	int f           = output_frame % output_size;
	int ix = f % output_w_size + fx;
	int iy = f / output_w_size + fy;

	int input_node  = (c * input_h_size  + iy) * input_w_size  + ix;
	float sig = in_sig[input_node * input_frame_size + input_frame];

	int output_node = (c * filter_h_size + fy) * filter_w_size + fx;	
	out_sig[output_node * output_frame_size + output_frame] = sig;
}


int cubb_Im2Col_Forward
		(
			const float*	dev_in_sig,
			float*			dev_out_sig,
			int				input_frame_size,
			int				input_w_size,
			int				input_h_size,
			int				input_c_size,
			int				filter_w_size,
			int				filter_h_size
		)
{
	int output_c_size = input_c_size;
	int output_w_size = input_w_size - filter_w_size + 1;
	int output_h_size = input_h_size - filter_h_size + 1;
	int output_size   = output_w_size * output_h_size;

//	int input_node_size  = input_c_size * input_h_size * input_w_size;
//	int output_node_size = output_c_size * filter_h_size * filter_w_size;
	
	int output_frame_size = input_frame_size * output_size;
	
	int		frame_unit = 16;
	dim3	grid(output_frame_size/frame_unit, output_c_size);
	dim3	block(frame_unit, filter_w_size, filter_h_size);
	
	kernal_Im2Col_Forward<<<grid, block>>>(
			dev_in_sig,
			dev_out_sig,
			input_frame_size,
			input_w_size,
			input_h_size,
			output_frame_size,
			output_w_size,
			output_size
		);
	BB_CUDA_CHECK_LAST_ERROR();

	return 0;
}



int bbcu_eva_Im2Col_Forward
		(
			const float*	in_sig,
			float*			out_sig,
			int				input_frame_size,
			int				input_w_size,
			int				input_h_size,
			int				input_c_size,
			int				filter_w_size,
			int				filter_h_size
		)
{
	cudaDeviceProp dev;
	BB_CUDA_SAFE_CALL(cudaGetDeviceProperties(&dev, 0));

	cudaError_t cudaStatus0 = cudaGetLastError();
    if (cudaStatus0 != cudaSuccess) {
        fprintf(stderr, "start failed: %s\n", cudaGetErrorString(cudaStatus0));
		exit(1);
    }

	cudaDeviceSynchronize();
	auto time0 = std::chrono::system_clock::now();

	int output_c_size = input_c_size;
	int output_w_size = input_w_size - filter_w_size + 1;
	int output_h_size = input_h_size - filter_h_size + 1;
	int output_size   = output_w_size * output_h_size;

	int input_node_size  = input_c_size * input_h_size * input_w_size;
	int output_node_size = output_c_size * filter_h_size * filter_w_size;
	
	int output_frame_size = input_frame_size * output_size;
	
	float* dev_in_sig;
	float* dev_out_sig;

	BB_CUDA_SAFE_CALL(cudaMalloc((void**)&dev_in_sig,   input_node_size  * input_frame_size * sizeof(float)));
	BB_CUDA_SAFE_CALL(cudaMalloc((void**)&dev_out_sig,  output_node_size * output_frame_size * sizeof(float)));
	
	cudaDeviceSynchronize();
	auto time1 = std::chrono::system_clock::now();

	BB_CUDA_SAFE_CALL(cudaMemcpy(dev_in_sig, in_sig, input_node_size * input_frame_size * sizeof(float), cudaMemcpyHostToDevice));

	cudaDeviceSynchronize();
	auto time2 = std::chrono::system_clock::now();
	
	int		frame_unit = 16;
	dim3	grid(output_frame_size/frame_unit, output_c_size);
	dim3	block(frame_unit, filter_w_size, filter_h_size);
	
	kernal_Im2Col_Forward<<<grid, block>>>(
			dev_in_sig,
			dev_out_sig,
			input_frame_size,
			input_w_size,
			input_h_size,
			output_frame_size,
			output_w_size,
			output_size
		);
	BB_CUDA_CHECK_LAST_ERROR();
	
	cudaDeviceSynchronize();
	auto time3 = std::chrono::system_clock::now();

	BB_CUDA_SAFE_CALL(cudaMemcpy(out_sig, dev_out_sig, output_node_size * output_frame_size * sizeof(float), cudaMemcpyDeviceToHost));

	cudaDeviceSynchronize();
	auto time4 = std::chrono::system_clock::now();

	BB_CUDA_SAFE_CALL(cudaFree(dev_in_sig));
	BB_CUDA_SAFE_CALL(cudaFree(dev_out_sig));

	cudaDeviceSynchronize();
	auto time5 = std::chrono::system_clock::now();

	double elapsed_malloc       = (double)std::chrono::duration_cast<std::chrono::milliseconds>(time1-time0).count();
	double elapsed_cpu_to_gpu   = (double)std::chrono::duration_cast<std::chrono::milliseconds>(time2-time1).count();
	double elapsed_kernel       = (double)std::chrono::duration_cast<std::chrono::milliseconds>(time3-time2).count();
	double elapsed_gpu_to_cpu   = (double)std::chrono::duration_cast<std::chrono::milliseconds>(time4-time3).count();
	double elapsed_free         = (double)std::chrono::duration_cast<std::chrono::milliseconds>(time5-time4).count();
	std::cout << "malloc               : " << elapsed_malloc       << " [msec]" << std::endl;
	std::cout << "data copy(cpu->gpu)  : " << elapsed_cpu_to_gpu   << " [msec]" << std::endl;
	std::cout << "kernel               : " << elapsed_kernel       << " [msec]" << std::endl;
	std::cout << "data copy(gpu->cpu)  : " << elapsed_gpu_to_cpu   << " [msec]" << std::endl;
	std::cout << "free                 : " << elapsed_free         << " [msec]" << std::endl;
	
	return 0;
}

