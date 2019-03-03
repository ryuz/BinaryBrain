#include <iostream>
#include <chrono>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bbcu/bbcu.h"
#include "bbcu/bbcu_util.h"



// kernel
__global__ void kernel_HorizontalSum(
			const float*	src,
			float*			dst,
			int				size)
{
	extern __shared__   float	buf[];

	// èâä˙âª
	int y      = blockIdx.x;
	int x      = threadIdx.x;
	int x_step = blockDim.x;

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
	while ( comb < size ) {
		int next = comb * 2;
		int mask = next - 1;
		if ( (x & mask) == 0 ) {
			buf[x] += buf[x + comb];
		}
		comb = next;
		__syncthreads();
	}
	
	dst[y] = buf[0];
}


int bbcu_HorizontalSum
		(
			const float*	dev_src,
			float*			dev_dst,
			int				x_size,
			int				y_size,
			cudaStream_t	streamId
		)
{
	int		unit_x = 512;

	dim3	grid(y_size);
	dim3	block(unit_x);
	
	kernel_HorizontalSum<<<grid, block, unit_x*sizeof(float), streamId>>>(
			dev_src,
			dev_dst,
			x_size);

	cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		exit(1);
		return 1;
    }
	
	return 0;
}



int bbcu_eva_HorizontalSum
		(
			const float*	src,
			float*			dst,
			int				x_size,
			int				y_size
		)
{
	float*	dev_src;
	float*	dev_dst;

	BB_CUDA_SAFE_CALL(cudaMalloc((void**)&dev_src, y_size * x_size * sizeof(float)));
	BB_CUDA_SAFE_CALL(cudaMalloc((void**)&dev_dst, y_size * sizeof(float)));

	BB_CUDA_SAFE_CALL(cudaMemcpy(dev_src, src, y_size * x_size * sizeof(float), cudaMemcpyHostToDevice));

	bbcu_HorizontalSum(dev_src, dev_dst, x_size, y_size, 0);

	BB_CUDA_SAFE_CALL(cudaMemcpy(dst, dev_dst, y_size * sizeof(float), cudaMemcpyDeviceToHost));

	return 0;
}

