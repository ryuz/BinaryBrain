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

	// 係数読み込み
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
	// 係数読み込み
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

	// 1つのSMで1nodeを全フレーム処理
	while ( frame <  frame_size ) {
		// 入力データ読み込み
		float	x[N];
		for ( int i = 0; i < N; ++i ) {
			x[i] = x_ptr[i][frame];
		}

		// 計算
		float sig1 = b1;
		for ( int i = 0; i < M; ++i ) {
			float sig0 = b0[i];
			for ( int j = 0; j < N; ++j ) {
				sig0 += x[j] * W0[i][j];
			}
		
			sig0 = fmaxf(sig0, 0);	// ReLU
		
			sig1 += sig0 * W1[i];
		}

		// 出力
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

__device__ __forceinline__ float device_fp32_LocalSum(float v, float *buf)
{
	buf[threadIdx.x] = v;
	__syncthreads();

	// スレッド間集計
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
template <int N=6, int M=16, int THREAD_SIZE=256>
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
	int const node    = blockIdx.x;
	int	const id      = threadIdx.x;
	int const id_step = THREAD_SIZE;

    __shared__  float sbuf[THREAD_SIZE];
	__shared__  float W0[M][N];
	__shared__  float b0[M];
	__shared__  float W1[M];


	// 係数読み込み
    for ( int i = id; i < M; i += id_step ) {
		for ( int j = 0; j < N; ++j ) {
			W0[i][j] = hidden_W[(node * M + i) * N + j];
		}

		b0[i] = hidden_b[node * M + i];
		W1[i] = output_W[node * M + i];
	}

    // ポインタ読み込み
    __shared__ float const *x_ptr[N];
	for ( int i = 0; i < N; ++i ) {
		int input_node = input_index[node*N + i];
		x_ptr[i] = &x_buf[frame_stride * input_node];
	}

	float const *dy_ptr = &dy_buf[frame_stride * node];

	__syncthreads();
	
	// 勾配初期化
 	float dW0[M][N];
	float db0[M];
	float dW1[M];
	float db1;
	for ( int i = 0; i < M; ++ i ) {
		for ( int j = 0; j < N; ++j ) {
			dW0[i][j] = 0;
		}
	}
	for ( int i = 0; i < M; ++i ) {
		db0[i] = 0;
		dW1[i] = 0;
	}
	db1 = 0;
    
	// 1つのSMで1nodeを全フレーム処理
    for ( int frame = id; frame < frame_size; frame += id_step ) {
		// 入力データ読み込み
		float	x[N];
		for ( int i = 0; i < N; ++i ) {
			x[i] = x_ptr[i][frame];
		}
		
		// 1段目再計算して2段目逆伝播
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
		
		// 1段目逆伝播
		float *dx_ptr  = &dx_buf[frame_stride * N * node];
		float	dx[N];
		for ( int i = 0; i < N; ++i ) {
			dx[i] = 0;	// dx_ptr[frame_stride * i + frame];
		}

		for ( int i = 0; i < M; ++i ) {
			db0[i] += grad0[i];
			for ( int j = 0; j < N; ++j ) {
				dW0[i][j] += grad0[i] * x[j];
				dx[j] += grad0[i] * W0[i][j];
			}
		}
		
		// 誤差書き込み
		for ( int i = 0; i < N; ++i ) {
			dx_ptr[frame_stride * i + frame] = dx[i];
		}
	}
	
	__syncthreads();

	// 係数統合
    for ( int i = 0; i < M; ++i ) {
		for ( int j = 0; j < N; ++j ) {
			dW0[i][j] = device_fp32_LocalSum(dW0[i][j], sbuf);
        }
    	db0[i] = device_fp32_LocalSum(db0[i], sbuf);
		dW1[i] = device_fp32_LocalSum(dW1[i], sbuf);
    }
   	db1 = device_fp32_LocalSum(db1, sbuf);

    // 勾配出力込み
    for ( int i = id; i < M; i += id_step ) {
		for ( int j = 0; j < N; ++j ) {
			hidden_dW[(node * M + i) * N + j] = dW0[i][j];
		}
		 hidden_db[node * M + i] = db0[i];
		 output_dW[node * M + i] = dW1[i];
	}
    if (id == 0) {
		 output_db[node] = db1;
    }
    __syncthreads();
}


template <int N=6>
__global__ void kernal_fp32_MicroMlp_BackwardMarge(
			float const *src_buf,
			float       *dst_buf,
			int   const *input_index,
			int			node_size,
			int			frame_size,
			int			frame_stride
		)
{
	int frame = blockDim.x * blockIdx.x + threadIdx.x;
	
	for ( int node = 0; node < node_size; ++node ) {
        if ( frame < frame_size ) {
    	    for ( int n = 0; n < N; ++n ) {
		        int in_idx = input_index[node*N + n];
		        float*		 dst_buf_ptr = &dst_buf[frame_stride * in_idx];
                float        prev_data   = dst_buf_ptr[frame];
		        const float* src_buf_ptr = &src_buf[(N * node + n) * frame_stride];
		        
		        dst_buf_ptr[frame] = prev_data + src_buf_ptr[frame];
            }
        }
		__syncthreads();
	}
}



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
        int const thread_size = 128;
        dim3    block(thread_size);
        dim3    grid(output_node_size);
        while ( frame_size < (int)block.x / 2 ) {
            block.x /= 2;
        }

        kernal_fp32_MicroMlp_Backward<N, M, thread_size><<<grid, block, 0, streamId>>>
            (
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

        int block_x = 1024;
        while ( block_x / 2 >= frame_size ) { block_x /= 2; }

        dim3    grid((frame_size + (block_x - 1)) / block_x);
        dim3    block(block_x);
        kernal_fp32_MicroMlp_BackwardMarge<N><<<grid, block>>>
            (
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
