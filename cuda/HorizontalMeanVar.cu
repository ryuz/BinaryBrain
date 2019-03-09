#include <iostream>
#include <chrono>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bbcu/bbcu.h"
#include "bbcu/bbcu_util.h"


// kernel
__global__ void kernel_fp32_HorizontalMeanVar(
	const float*	src,
	float*			mean,
	float*			variance,
	int				frame_size,
	int				frame_stride)
{
	extern __shared__   float	buf[];
	
	// ������
	int node  = blockIdx.x;
	int frame = threadIdx.x;
	int frame_step = blockDim.x;
	
	
	// �J�n���̉��Z�A���S���Y��(Kahan summation algorithm)
	float s1 = 0, c1 = 0, y1, t1;
	float s1 = 0, c1 = 0, y1, t1;
	const float* src_ptr = &src[frame_stride * node];
	while ( frame < frame_size ) {
		float x = src_ptr[frame];
		
		y1 = x - c1;
		t1 = s1 + y1;
		c1 += (t1 - s1) - y1;
		
		y2 = x - c2;
		t2 = s2 + y2;
		c2 += (t2 - s2) - y2;
		
		frame += frame_step;
	}
	
	float* buf1 = &buf[0];
	float* buf2 = &buf[blockDim.x];
	
	buf1[threadIdx.x] = s1;
	buf2[threadIdx.x] = s2;
	
	__syncthreads();
	
	// �X���b�h�ԏW�v
	int comb = 1;
	while (comb < frame_size) {
		int next = comb * 2;
		int mask = next - 1;
		if ((threadIdx.x & mask) == 0) {
			buf1[threadIdx.x] += buf1[threadIdx.x + comb];
			buf2[threadIdx.x] += buf2[threadIdx.x + comb];
		}
		comb = next;
		__syncthreads();
	}
	
	float m = buf1[0] / frame_size;
	float v = (buf2[0] / frame_size) - (m * m);
	
	mean[node]     = m;
	variance[node] = v;
}


int bbcu_fp32_HorizontalMeanVar
(
	const float*	dev_src,
	float*			dev_mean,
	float*			dev_var,
	int				node_size,
	int				frame_size,
	int				frame_stride,
	cudaStream_t	streamId
)
{
	BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());
	
	int		unit_x = 512;
	
	dim3	grid(node_size);
	dim3	block(unit_x);
	
	kernel_fp32_HorizontalMeanVar << <grid, block, 2 * unit_x * sizeof(float), streamId >> > (
		dev_src,
		dev_dst,
		frame_size,
		frame_stride);
	
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		exit(1);
		return 1;
	}

	return 0;
}

