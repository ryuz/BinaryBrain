#include <iostream>
#include <chrono>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bbcu/bbcu.h"
#include "bbcu/bbcu_util.h"



// -------------------------------------------------
//  Forward
// -------------------------------------------------

#define FORWARD_USE_SHARED_MEM  0

template <int N=6, int M=16>
__global__ void kernal_fp32_MicroMlp_Forward(
			const float*	x_buf,
			float*			y_buf,
			const int*		input_index,
			const float*	hidden_W,
			const float*	hidden_b,
			const float*	output_W,
			const float*	output_b,
   			int				frame_size,
   			int				frame_stride
        )
{
	int frame_step = blockDim.x;
	int frame      = threadIdx.x;
	int node       = blockIdx.x;

#if FORWARD_USE_SHARED_MEM
	__shared__   float W0[M][N];
	__shared__   float b0[M];
	__shared__   float W1[M];
	__shared__	 float b1;

	// åWêîì«Ç›çûÇ›
	if (threadIdx.x < M) {
		int i = threadIdx.x;

		for ( int j = 0; j < N; ++j ) {
			W0[i][j] = hidden_W[(node * M + i) * N + j];
		}

		b0[i] = hidden_b[node * M + i];
		W1[i] = output_W[node * M + i];
	}
	if (threadIdx.x == 0) {
		b1 = output_b[node];
	}

	__syncthreads();
#else
	float W0[M][N];
	float b0[M];
	float W1[M];
	float b1;
	// åWêîì«Ç›çûÇ›
	for ( int i = 0; i < M; ++i ) {
		for ( int j = 0; j < N; ++j ) {
			W0[i][j] = hidden_W[(node * M + i) * N + j];
		}

		b0[i] = hidden_b[node * M + i];
		W1[i] = output_W[node * M + i];
	}
	b1 = output_b[node];
#endif

	const float *x_ptr[N];
	for ( int i = 0; i < N; ++i ) {
		int in_idx = input_index[node*N + i];
		x_ptr[i] = &x_buf[frame_stride * in_idx];
	}

	float *y_ptr = &y_buf[frame_stride * node];

	// 1Ç¬ÇÃSMÇ≈1nodeÇëSÉtÉåÅ[ÉÄèàóù
	while ( frame <  frame_size ) {
		// ì¸óÕÉfÅ[É^ì«Ç›çûÇ›
		float	x[N];
		for ( int i = 0; i < N; ++i ) {
			x[i] = x_ptr[i][frame];
		}

		// åvéZ
		float sig1 = b1;
		for ( int i = 0; i < M; ++i ) {
			float sig0 = b0[i];
			for ( int j = 0; j < N; ++j ) {
				sig0 += x[j] * W0[i][j];
			}
		
			sig0 = fmaxf(sig0, 0);	// ReLU
		
			sig1 += sig0 * W1[i];
		}

		// èoóÕ
		y_ptr[frame] = sig1;

		frame += frame_step;
	}
}


template <int N=6, int M=16>
int bbcu_fp32_MicroMlp_Forward
		(
			const float*	dev_x_buf,
			float*			dev_y_buf,
			const int*		dev_input_index,
			const float*	dev_hidden_W,
			const float*	dev_hidden_b,
			const float*	dev_output_W,
			const float*	dev_output_b,
			int				input_node_size,
			int				output_node_size,
			int				frame_size,
			int				frame_stride,
			cudaStream_t	streamId = 0
		)
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

	dim3	grid(output_node_size);
#if FORWARD_USE_SHARED_MEM
	dim3	block(512, 1, 1);
#else
    dim3	block(192, 1, 1);
#endif

	kernal_fp32_MicroMlp_Forward<N, M><<<grid, block, 0, streamId>>>(
			dev_x_buf,
			dev_y_buf,
			dev_input_index,
			dev_hidden_W,
			dev_hidden_b,
			dev_output_W,
			dev_output_b,
			frame_size,
			frame_stride
		);
	BB_CUDA_CHECK_LAST_ERROR();
	
	return 0;
}


int bbcu_fp32_MicroMlp6x16_Forward
		(
			float const     *dev_x_buf,
			float           *dev_y_buf,
			int   const     *dev_input_index,
			float const     *dev_hidden_W,
			float const     *dev_hidden_b,
			float const     *dev_output_W,
			float const     *dev_output_b,
			int			    input_node_size,
			int			    output_node_size,
			int			    frame_size,
			int			    frame_stride,
			cudaStream_t    streamId
		)
{
	return bbcu_fp32_MicroMlp_Forward<6, 16>(
			dev_x_buf,
			dev_y_buf,
			dev_input_index,
			dev_hidden_W,
			dev_hidden_b,
			dev_output_W,
			dev_output_b,
			input_node_size,
			output_node_size,
			frame_size,
            frame_stride,
			streamId
		);
}





// -------------------------------------------------
//  Backward
// -------------------------------------------------

#define BACKWARD_USE_SHARED_MEM  0


#if BACKWARD_USE_SHARED_MEM

// kernel
template <int N=6, int M=16, int H=16>
__global__ void kernal_fp32_MicroMlp_Backward
        (
			float const     *x_buf,
			float const     *dy_buf,
			float           *dx_buf,
			int   const     *input_index,
			float const     *hidden_W,
			float const     *hidden_b,
			float           *hidden_dW,
			float           *hidden_db,
			float const     *output_W,
			float const     *output_b,
			float           *output_dW,
			float           *output_db,
			int				frame_size,
			int				frame_stride
        )
{
	int	id         = threadIdx.x;
	int frame_step = H;	// blockDim.x;
	int frame      = threadIdx.x;
	int node       = blockIdx.x;

	__shared__   float W0[M][N];
	__shared__   float b0[M];
	__shared__   float W1[M];
//				 float b1;

 	__shared__   float dW0[M][N][H];
	__shared__   float db0[M][H];
	__shared__   float dW1[M][H];
	__shared__	 float db1[H];

	// åWêîì«Ç›çûÇ›
	if (threadIdx.x < M) {
		int i = threadIdx.x;

		for ( int j = 0; j < N; ++j ) {
			W0[i][j] = hidden_W[(node * M + i) * N + j];
		}

		b0[i] = hidden_b[node * M + i];
		W1[i] = output_W[node * M + i];
	}
//	if (threadIdx.x == 0) {
//		b1 = output_b[node];
//	}
	
	// å˘îzèâä˙âª
	for ( int i = 0; i < M; ++ i ) {
		for ( int j = 0; j < N; ++j ) {
			dW0[i][j][id] = 0; // hidden_dW[(node * M + i) * N + j];
		}
	}
	for ( int i = 0; i < M; ++i ) {
		db0[i][id] = 0; // hidden_db[node * M + i];
	}
	for ( int i = 0; i < M; ++i ) {
		dW1[i][id] = 0; // output_dW[node * M + i];
	}
	db1[id] = 0; // output_db[node];

	__syncthreads();

	const float *x_ptr[N];
	for ( int i = 0; i < N; ++i ) {
		int in_idx = input_index[node*N + i];
		x_ptr[i] = &x_buf[frame_stride * in_idx];
	}

	float const *dy_ptr = &dy_buf[frame_stride * node];

	// 1Ç¬ÇÃSMÇ≈1nodeÇëSÉtÉåÅ[ÉÄèàóù
    frame = threadIdx.x;
	while ( frame < frame_size ) {
		// ì¸óÕÉfÅ[É^ì«Ç›çûÇ›
		float	x[N];
		for ( int i = 0; i < N; ++i ) {
			x[i] = x_ptr[i][frame];
		}
		
		// 1íiñ⁄çƒåvéZÇµÇƒ2íiñ⁄ãtì`îd
		float	grad1 = dy_ptr[frame];
		float	grad0[M];
		db1[id] += grad1;
		for ( int i = 0; i < M; ++i ) {
			float sig0 = b0[i];
			for ( int j = 0; j < N; ++j ) {
				sig0 += x[j] * W0[i][j];
			}
		
			sig0 = fmaxf(sig0, 0);	// ReLU

			dW1[i][id] += grad1 * sig0;

			if ( sig0 > 0 ) {		// ReLU
				grad0[i] = grad1 * W1[i];
			}
			else {
				grad0[i] = 0;
			}
		}
		
		// 1íiñ⁄ãtì`îd
		float *dx_ptr  = &dx_buf[frame_stride * N * node];
		float	dx[N];
		for ( int i = 0; i < N; ++i ) {
			dx[i] = 0;	// dx_ptr[frame_stride * i + frame];
		}

		for ( int i = 0; i < M; ++i ) {
			db0[i][id] += grad0[i];
			for ( int j = 0; j < N; ++j ) {
				dW0[i][j][id] += grad0[i] * x[j];
				dx[j] += grad0[i] * W0[i][j];
			}
		}
		
		// åÎç∑èëÇ´çûÇ›
		for ( int i = 0; i < N; ++i ) {
			dx_ptr[frame_stride * i + frame] = dx[i];
		}

		frame += frame_step;
	}
	
	__syncthreads();

	int comb = 1;
	while ( comb < H ) {
		int next = comb * 2;
		int mask = next - 1;
		if ( (threadIdx.x & mask) == 0 && id + comb < H ) {
			for ( int i = 0; i < M; ++ i ) {
				for ( int j = 0; j < N; ++j ) {
					dW0[i][j][id] += dW0[i][j][id + comb];
				}
			}
			for ( int i = 0; i < M; ++i ) {
				db0[i][id] += db0[i][id + comb];
			}
			for ( int i = 0; i < M; ++i ) {
				dW1[i][id] += dW1[i][id + comb];
			}
			db1[id] += db1[id + comb];
		}
		comb = next;
		__syncthreads();
	}

	// å˘îzèoóÕ(å„Ç≈ï¿óÒâªÇ∑ÇÈ)
	if ( threadIdx.x == 0 ) {
		for ( int i = 0; i < M; ++i ) {
			for ( int j = 0; j < N; ++j ) {
				hidden_dW[(node * M + i) * N + j] = dW0[i][j][0];
			}
		}
		for ( int i = 0; i < M; ++i ) {
			hidden_db[node * M + i] = db0[i][0];
		}
		for ( int i = 0; i < M; ++i ) {
			output_dW[node * M + i] = dW1[i][0];
		}
		output_db[node] = db1[0];
	}
}

#else


__device__ __forceinline__ float device_fp32_LocalSum(float v, float *buf)
{
	buf[threadIdx.x] = v;
	__syncthreads();

	// ÉXÉåÉbÉhä‘èWåv
	int comb = 1;
	while (comb < blockDim.x) {
		int next = comb * 2;
		int mask = next - 1;
		if ((threadIdx.x & mask) == 0) {
			buf[threadIdx.x] += buf[threadIdx.x + comb];
		}
		comb = next;
		__syncthreads();
	}

    float sum = buf[0];
    __syncthreads();
	
    return sum;
}


// kernel
template <int N=6, int M=16, int H=16>
__global__ void kernal_fp32_MicroMlp_Backward
        (
			float const     *x_buf,
			float const     *dy_buf,
			float           *dx_buf,
			int   const     *input_index,
			float const     *hidden_W,
			float const     *hidden_b,
			float           *hidden_dW,
			float           *hidden_db,
			float const     *output_W,
			float const     *output_b,
			float           *output_dW,
			float           *output_db,
			int				frame_size,
			int				frame_stride
        )
{
	int frame_step = H;	// blockDim.x;
	int frame_base = threadIdx.x;
	int node       = blockIdx.x;

    __shared__ float buf[H];

	float W0[M][N];
	float b0[M];
	float W1[M];
//	float b1;

 	float dW0[M][N];
	float db0[M];
	float dW1[M];
	float db1;

	// åWêîì«Ç›çûÇ›
	for ( int i = 0; i < M; ++i ) {
		for ( int j = 0; j < N; ++j ) {
			W0[i][j] = hidden_W[(node * M + i) * N + j];
		}

		b0[i] = hidden_b[node * M + i];
		W1[i] = output_W[node * M + i];
	}
//	b1 = output_b[node];
	
	// å˘îzèâä˙âª
	for ( int i = 0; i < M; ++ i ) {
		for ( int j = 0; j < N; ++j ) {
			dW0[i][j] = 0;
		}
	}
	for ( int i = 0; i < M; ++i ) {
		db0[i] = 0;
	}
	for ( int i = 0; i < M; ++i ) {
		dW1[i] = 0;
	}
	db1 = 0;

	const float *x_ptr[N];
	for ( int i = 0; i < N; ++i ) {
		int in_idx = input_index[node*N + i];
		x_ptr[i] = &x_buf[frame_stride * in_idx];
	}

	float const *dy_ptr = &dy_buf[frame_stride * node];

	// 1Ç¬ÇÃSMÇ≈1nodeÇëSÉtÉåÅ[ÉÄèàóù
	for ( int frame = frame_base; frame < frame_size; frame += frame_step ) {
		// ì¸óÕÉfÅ[É^ì«Ç›çûÇ›
		float	x[N];
		for ( int i = 0; i < N; ++i ) {
			x[i] = x_ptr[i][frame];
		}
		
		// 1íiñ⁄çƒåvéZÇµÇƒ2íiñ⁄ãtì`îd
		float	grad1 = dy_ptr[frame];
		float	grad0[M];
		db1 += grad1;
		for ( int i = 0; i < M; ++i ) {
			float sig0 = b0[i];
			for ( int j = 0; j < N; ++j ) {
				sig0 += x[j] * W0[i][j];
			}
		
			sig0 = fmaxf(sig0, 0);	// ReLU

			dW1[i] += grad1 * sig0;

			if ( sig0 > 0 ) {		// ReLU
				grad0[i] = grad1 * W1[i];
			}
			else {
				grad0[i] = 0;
			}
		}
		
		// 1íiñ⁄ãtì`îd
		float *dx_ptr  = &dx_buf[frame_stride * N * node];
		float	dx[N];
		for ( int i = 0; i < N; ++i ) {
			dx[i] = 0;	// dx_ptr[frame_stride * i + frame];
		}

		for ( int i = 0; i < M; ++i ) {
			db0[i] += grad0[i];
			for ( int j = 0; j < N; ++j ) {
				dW0[i][j] += grad0[i] * x[j];
				dx[j]     += grad0[i] * W0[i][j];
			}
		}
		
		// åÎç∑èëÇ´çûÇ›
		for ( int i = 0; i < N; ++i ) {
			dx_ptr[frame_stride * i + frame] = dx[i];
		}
	}
	
	for ( int i = 0; i < M; ++ i ) {
		for ( int j = 0; j < N; ++j ) {
			dW0[i][j] = device_fp32_LocalSum(dW0[i][j], buf);
		}
	}
	for ( int i = 0; i < M; ++i ) {
		db0[i] = device_fp32_LocalSum(db0[i], buf);
	}
	for ( int i = 0; i < M; ++i ) {
		dW1[i] = device_fp32_LocalSum(dW1[i], buf);
	}
	db1 = device_fp32_LocalSum(db1, buf);
    
	// å˘îzèoóÕ(å„Ç≈ï¿óÒâªÇ∑ÇÈ)
    if (H >= M) {
        int i = threadIdx.x;
        if ( i < M ) {
	        for ( int j = 0; j < N; ++j ) {
		        hidden_dW[(node * M + i) * N + j] = dW0[i][j];
	        }
	        hidden_db[node * M + i] = db0[i];
	        output_dW[node * M + i] = dW1[i];
            if ( i == 0 ) {
		        output_db[node] = db1;
	        }
        }
    }
    else {
    	if ( threadIdx.x == 0 ) {
		    for ( int i = 0; i < M; ++i ) {
			    for ( int j = 0; j < N; ++j ) {
				    hidden_dW[(node * M + i) * N + j] = dW0[i][j];
			    }
		    }
		    for ( int i = 0; i < M; ++i ) {
			    hidden_db[node * M + i] = db0[i];
		    }
		    for ( int i = 0; i < M; ++i ) {
			    output_dW[node * M + i] = dW1[i];
		    }
		    output_db[node] = db1;
	    }
    }
}
#endif


#if 1
template <int N=6>
__global__ void kernal_fp32_MicroMlp_BackwardMarge(
			const float*	src_buf,
			float*			dst_buf,
			const int*		input_index,
			int				node_size,
			int				frame_size,
			int				frame_stride
		)
{
	int n          = blockDim.y * blockIdx.y + threadIdx.y;
	int frame      = blockDim.x * blockIdx.x + threadIdx.x;
	
	for ( int node = 0; node < node_size; ++node ) {
		int in_idx = input_index[node*N + n];
		float*		 dst_buf_ptr = &dst_buf[frame_size * in_idx];
		const float* src_buf_ptr = &src_buf[(N * node + n) * frame_size];
		
		dst_buf_ptr[frame] += src_buf_ptr[frame];

		__syncthreads();
	}
}
#else

template <int N=6>
__global__ void kernal_fp32_MicroMlp_BackwardMarge(
			const float*	src_buf,
			float*			dst_buf,
			const int*		input_index,
			int				node_size,
			int				frame_size,
			int				frame_stride
		)
{
	int n          = blockDim.y * blockIdx.y + threadIdx.y;
	int frame_base = threadIdx.x;
	int frame_step = blockDim.x;
	
	for ( int node = 0; node < node_size; ++node ) {
		int in_idx = input_index[node*N + n];
		float*		 dst_buf_ptr = &dst_buf[frame_size * in_idx];
		const float* src_buf_ptr = &src_buf[(N * node + n) * frame_size];
    	for ( int frame = frame_base; frame < frame_size; frame += frame_step ) {
    		dst_buf_ptr[frame] += src_buf_ptr[frame];
        }
	}
}

#endif


template <int N=6, int M=16>
int bbcu_fp32_MicroMlp_Backward(
			float const     *dev_x_buf,
			float const     *dev_dy_buf,
			float           *dev_dx_buf,
			float           *dev_dx_tmp,
			int   const     *dev_input_index,
			float const     *dev_hidden_W,
			float const     *dev_hidden_b,
			float           *dev_hidden_dW,
			float           *dev_hidden_db,
			float const     *dev_output_W,
			float const     *dev_output_b,
			float           *dev_output_dW,
			float           *dev_output_db,
			int				input_node_size,
			int				output_node_size,
			int				frame_size,
			int				frame_stride,
			cudaStream_t	streamId = 0
	)
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

	{
		const int x_size = 128; // (8192 / (N*M));

		dim3	grid(output_node_size);
		dim3	block(x_size, 1, 1);
		
		kernal_fp32_MicroMlp_Backward<N, M, x_size><<<grid, block, 0, streamId>>>(
				dev_x_buf,
				dev_dy_buf,
				dev_dx_tmp,
				dev_input_index,
				dev_hidden_W,
				dev_hidden_b,
				dev_hidden_dW,
				dev_hidden_db,
				dev_output_W,
				dev_output_b,
				dev_output_dW,
				dev_output_db,
				frame_size,
				frame_stride
			);
        BB_CUDA_CHECK_LAST_ERROR();
	}

    {
        BB_CUDA_SAFE_CALL(cudaMemset(dev_dx_buf, 0, input_node_size * frame_stride * sizeof(float)));

#if 1
	    int block_x = frame_size;
	    while ( block_x > 1024 ) { block_x /= 2; }

	    dim3	grid((frame_size + block_x - 1) /block_x, N);
	    dim3	block(block_x, 1, 1);
#else
	    dim3	grid(1, N);
	    dim3	block(1024, 1, 1);
#endif

	    kernal_fp32_MicroMlp_BackwardMarge<N><<<grid, block>>>(
			    dev_dx_tmp,
			    dev_dx_buf,
			    dev_input_index,
			    output_node_size,
			    frame_size,
			    frame_stride
		    );
        BB_CUDA_CHECK_LAST_ERROR();
    }

	return 0;
}


BBCU_DLL_EXPORT int bbcu_fp32_MicroMlp6x16_Backward(
			float const     *dev_x_buf,
			float const     *dev_dy_buf,
			float           *dev_dx_buf,
			float           *dev_dx_tmp,
			int   const     *dev_input_index,
			float const     *dev_hidden_W,
			float const     *dev_hidden_b,
			float           *dev_hidden_dW,
			float           *dev_hidden_db,
			float const     *dev_output_W,
			float const     *dev_output_b,
			float           *dev_output_dW,
			float           *dev_output_db,
			int				input_node_size,
			int				output_node_size,
			int				frame_size,
			int				frame_stride,
			cudaStream_t	streamId
		)
{
	return bbcu_fp32_MicroMlp_Backward<6, 16>(
			dev_x_buf,
			dev_dy_buf,
			dev_dx_buf,
			dev_dx_tmp,
			dev_input_index,
			dev_hidden_W,
			dev_hidden_b,
			dev_hidden_dW,
			dev_hidden_db,
			dev_output_W,
			dev_output_b,
			dev_output_dW,
			dev_output_db,
			input_node_size,
			output_node_size,
			frame_size,
			frame_stride,
			streamId
		);
}



// end of file
