#include <iostream>
#include <chrono>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bbcu/bbcu.h"
#include "bbcu/bbcu_util.h"



//////////////////////////////
// forward
//////////////////////////////

__global__ void kernal_fp32_BatchNormalization_Forward(
			const float*	x_buf,
			float*			y_buf,
			float*			gamma_buf,
			float*			beta_buf,
			float*			mean_buf,
			float*			rstd_buf,
			float*			running_mean_buf,
			float*			running_var_buf,
			float			momentum,
			float			reciprocal_frame_size,
			int				frame_size,
			int				frame_stride
		)
{
	extern __shared__   float	buf[];

	// 初期化
	int node = blockIdx.x;
	int frame = threadIdx.x;
	int frame_step = blockDim.x;
	
	// カハンの加算アルゴリズム(Kahan summation algorithm)
	float s1 = 0, c1 = 0, y1, t1;
	float s2 = 0, c2 = 0, y2, t2;
	const float* x_ptr = &x_buf[frame_stride * node];
	while (frame < frame_size) {
		float x = x_ptr[frame];

		y1 = x - c1;
		t1 = s1 + y1;
		c1 += (t1 - s1) - y1;
		s1 = t1;

		y2 = (x * x) - c2;
		t2 = s2 + y2;
		c2 += (t2 - s2) - y2;
		s2 = t2;

		frame += frame_step;
	}

	float* buf1 = &buf[0];
	float* buf2 = &buf[blockDim.x];

	buf1[threadIdx.x] = s1;
	buf2[threadIdx.x] = s2;

	__syncthreads();

	// スレッド間集計
	int comb = 1;
	while (comb < frame_step) {
		int next = comb * 2;
		int mask = next - 1;
		if ((threadIdx.x & mask) == 0) {
			buf1[threadIdx.x] += buf1[threadIdx.x + comb];
			buf2[threadIdx.x] += buf2[threadIdx.x + comb];
		}
		comb = next;
		__syncthreads();
	}

	float mean;
	float var;
	float rstd;
	if (threadIdx.x == 0) {
		mean = buf1[0] * reciprocal_frame_size;
		var = max(10e-7f, (buf2[0] * reciprocal_frame_size) - (mean * mean));
		running_mean_buf[node] = running_mean_buf[node] * momentum + mean * (1.0f - momentum);
		running_var_buf[node] = running_var_buf[node] * momentum + var * (1.0f - momentum);
		rstd    = rsqrt(var);
		buf1[0] = mean;
		buf1[1] = var;
		buf1[2] = rstd;
		mean_buf[node] = mean;
		rstd_buf[node] = rstd;
	}
	__syncthreads();
	mean = buf1[0];
	var  = buf1[1];
	rstd = buf1[2];

	float gamma = gamma_buf[node];
	float beta  = beta_buf[node];
	float* y_ptr = &y_buf[frame_stride * node];
	frame = threadIdx.x;
	while (frame < frame_size) {
		float x = x_ptr[frame];
		x = (x - mean) * rstd;
		x = x * gamma + beta;
		y_ptr[frame] = x;
		frame += frame_step;
	}
}


CUBB_DLL_EXPORT int cubb_fp32_BatchNormalization_Forward
		(
			const float*	dev_x_buf,
			float*			dev_y_buf,
			float*			dev_gamma_buf,
			float*			dev_beta_buf,
			float*			dev_mean_buf,
			float*			dev_rstd_buf,
			float*			dev_running_mean_buf,
			float*			dev_running_var_buf,
			float			momentum,
			int				frame_size,
			int				frame_stride,
			int				node_size,	
			cudaStream_t    streamId
        )
{
	BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

	int		unit_x = 512;

	dim3	grid(node_size);
	dim3	block(unit_x);

	kernal_fp32_BatchNormalization_Forward<<<grid, block, 2 * unit_x * sizeof(float), streamId>>> (
			dev_x_buf,
            dev_y_buf,
			dev_gamma_buf,
			dev_beta_buf,
			dev_mean_buf,
			dev_rstd_buf,
			dev_running_mean_buf,
			dev_running_var_buf,
			momentum,
			1.0f/ frame_size,
			frame_size,
			frame_stride
		);
	BB_CUDA_CHECK_LAST_ERROR();

	return 0;
}


#if 0

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