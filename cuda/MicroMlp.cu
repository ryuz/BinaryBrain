#include <iostream>
#include <chrono>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bbcu/bbcu.h"
#include "bbcu/bbcu_util.h"



// -------------------------------------------------
//  Forward
// -------------------------------------------------

template <int N=6, int M=16>
__global__ void kernal_MicroMlp_Forward(
			const float*	in_sig_buf,
			float*			out_sig_buf,
			int				frame_size,
			const int*		input_index,
			const float*	hidden_W,
			const float*	hidden_b,
			const float*	output_W,
			const float*	output_b)
{
	int frame_step = blockDim.x;
	int frame      = threadIdx.x;
	int node       = blockIdx.x;

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

	const float *in_sig_ptr[N];
	for ( int i = 0; i < N; ++i ) {
		int in_idx = input_index[node*N + i];
		in_sig_ptr[i] = &in_sig_buf[frame_size * in_idx];
	}

	float *out_sig_ptr = &out_sig_buf[frame_size * node];

	// 1Ç¬ÇÃSMÇ≈1nodeÇëSÉtÉåÅ[ÉÄèàóù
	while ( frame <  frame_size ) {
		// ì¸óÕÉfÅ[É^ì«Ç›çûÇ›
		float	in_sig[N];
		for ( int i = 0; i < N; ++i ) {
			in_sig[i] = in_sig_ptr[i][frame];
		}

		// åvéZ
		float sig1 = b1;
		for ( int i = 0; i < M; ++i ) {
			float sig0 = b0[i];
			for ( int j = 0; j < N; ++j ) {
				sig0 += in_sig[j] * W0[i][j];
			}
		
			sig0 = fmaxf(sig0, 0);	// ReLU
		
			sig1 += sig0 * W1[i];
		}

		// èoóÕ
		out_sig_ptr[frame] = sig1;

		frame += frame_step;
	}
}


template <int N=6, int M=16>
int bbcu_MicroMlp_Forward
		(
			const float*	dev_in_sig,
			float*			dev_out_sig,
			int				input_node_size,
			int				output_node_size,
			int				frame_size,
			const int*		dev_input_index,
			const float*	dev_hidden_W,
			const float*	dev_hidden_b,
			const float*	dev_output_W,
			const float*	dev_output_b,
			cudaStream_t	streamId = 0
		)
{
	dim3	grid(output_node_size);
	dim3	block(512, 1, 1);
	
	kernal_MicroMlp_Forward<N, M><<<grid, block, 0, streamId>>>(
			dev_in_sig,
			dev_out_sig,
			frame_size,
			dev_input_index,
			dev_hidden_W,
			dev_hidden_b,
			dev_output_W,
			dev_output_b
		);
	BB_CUDA_CHECK_LAST_ERROR();
	
	return 0;
}


int bbcu_MicroMlp6x16_Forward
		(
			float const*	dev_in_sig,
			float*	        dev_out_sig,
			int				input_node_size,
			int				output_node_size,
			int				frame_size,
			const int*		dev_input_index,
			const float*	dev_hidden_W,
			const float*	dev_hidden_b,
			const float*	dev_output_W,
			const float*	dev_output_b,
			cudaStream_t	streamId
		)
{
	return bbcu_MicroMlp_Forward<6, 16>(
			dev_in_sig,
			dev_out_sig,
			input_node_size,
			output_node_size,
			frame_size,
			dev_input_index,
			dev_hidden_W,
			dev_hidden_b,
			dev_output_W,
			dev_output_b,
			streamId
		);
}





// -------------------------------------------------
//  Backward
// -------------------------------------------------

// kernel
template <int N=6, int M=16, int H=16>
__global__ void kernal_MicroMlp_Backward(
			const float*	in_sig_buf,
			float*			in_err_buf,
			float*			out_err_buf,
			int				frame_size,
			const int*		input_index,
			const float*	hidden_W,
			const float*	hidden_b,
			float*			hidden_dW,
			float*			hidden_db,
			const float*	output_W,
			const float*	output_b,
			float*			output_dW,
			float*			output_db)
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

	const float *in_sig_ptr[N];
	for ( int i = 0; i < N; ++i ) {
		int in_idx = input_index[node*N + i];
		in_sig_ptr[i] = &in_sig_buf[frame_size * in_idx];
	}

	float	*out_err_ptr = &out_err_buf[frame_size * node];

	// 1Ç¬ÇÃSMÇ≈1nodeÇëSÉtÉåÅ[ÉÄèàóù
	while ( frame <  frame_size ) {
		// ì¸óÕÉfÅ[É^ì«Ç›çûÇ›
		float	in_sig[N];
		for ( int i = 0; i < N; ++i ) {
			in_sig[i] = in_sig_ptr[i][frame];
		}
		
		// 1íiñ⁄çƒåvéZÇµÇƒ2íiñ⁄ãtì`îd
		float	err1 = out_err_ptr[frame];
		float	err0[M];
		db1[id] += err1;
		for ( int i = 0; i < M; ++i ) {
			float sig0 = b0[i];
			for ( int j = 0; j < N; ++j ) {
				sig0 += in_sig[j] * W0[i][j];
			}
		
			sig0 = fmaxf(sig0, 0);	// ReLU

			dW1[i][id] += err1 * sig0;

			if ( sig0 > 0 ) {		// ReLU
				err0[i] = err1 * W1[i];
			}
			else {
				err0[i] = 0;
			}
		}
		
		// 1íiñ⁄ãtì`îd
		float *in_err_ptr  = &in_err_buf[frame_size * N * node];
		float	in_err[N];
		for ( int i = 0; i < N; ++i ) {
			in_err[i] = 0;	// in_err_ptr[frame_size * i + frame];
		}

		for ( int i = 0; i < M; ++i ) {
			db0[i][id] += err0[i];
			for ( int j = 0; j < N; ++j ) {
				dW0[i][j][id] += err0[i] * in_sig[j];
				in_err[j] += err0[i] * W0[i][j];
			}
		}
		
		// åÎç∑èëÇ´çûÇ›
		for ( int i = 0; i < N; ++i ) {
			in_err_ptr[frame_size * i + frame] = in_err[i];
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


template <int N=6>
__global__ void kernal_MicroMlp_BackwardMarge(
			float*			dst_buf,
			const float*	src_buf,
			int				frame_size,
			int				node_size,
			const int*		input_index
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


template <int N=6, int M=16>
int bbcu_MicroMlp_Backward(
			const float*	dev_in_sig_buf,
			float*			dev_in_err_buf,
			float*			dev_in_err_tmp,
			float*			dev_out_err_buf,
			int				input_node_size,
			int				output_node_size,
			int				frame_size,
			const int*		dev_input_index,
			const float*	dev_hidden_W,
			const float*	dev_hidden_b,
			float*			dev_hidden_dW,
			float*			dev_hidden_db,
			const float*	dev_output_W,
			const float*	dev_output_b,
			float*			dev_output_dW,
			float*			dev_output_db,
			cudaStream_t	streamId = 0
	)
{
	{
		const int x_size = (8192 / (N*M));

		dim3	grid(output_node_size);
		dim3	block(x_size, 1, 1);
		
		kernal_MicroMlp_Backward<N, M, x_size><<<grid, block, 0, streamId>>>(
				dev_in_sig_buf,
				dev_in_err_tmp,
				dev_out_err_buf,
				frame_size,
				dev_input_index,
				dev_hidden_W,
				dev_hidden_b,
				dev_hidden_dW,
				dev_hidden_db,
				dev_output_W,
				dev_output_b,
				dev_output_dW,
				dev_output_db
			);

		cudaError_t cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			return 1;
		}
	}

    {
        BB_CUDA_SAFE_CALL(cudaMemset(dev_in_err_buf, 0, input_node_size * frame_size * sizeof(float)));

	    int block_x = frame_size;
	    while ( block_x > 1024 ) { block_x /= 2; }

	    dim3	grid(frame_size/block_x, N);
	    dim3	block(block_x, 1, 1);

	    kernal_MicroMlp_BackwardMarge<N><<<grid, block>>>(
			    dev_in_err_buf,
			    dev_in_err_tmp,
			    frame_size,
			    output_node_size,
			    dev_input_index
		    );

	    cudaError_t cudaStatus = cudaGetLastError();
	    if (cudaStatus != cudaSuccess) {
		    fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		    return 1;
	    }
    }

	return 0;
}


CUBB_DLL_EXPORT int bbcu_MicroMlp6x16_Backward(
			const float*	dev_in_sig_buf,
			float*			dev_in_err_buf,
			float*			dev_in_err_tmp,
			float*			dev_out_err_buf,
			int				input_node_size,
			int				output_node_size,
			int				frame_size,
			const int*		dev_input_index,
			const float*	dev_hidden_W,
			const float*	dev_hidden_b,
			float*			dev_hidden_dW,
			float*			dev_hidden_db,
			const float*	dev_output_W,
			const float*	dev_output_b,
			float*			dev_output_dW,
			float*			dev_output_db,
			cudaStream_t	streamId
		)
{
	return bbcu_MicroMlp_Backward<6, 16>(
			dev_in_sig_buf,
			dev_in_err_buf,
			dev_in_err_tmp,
			dev_out_err_buf,
			input_node_size,
			output_node_size,
			frame_size,
			dev_input_index,
			dev_hidden_W,
			dev_hidden_b,
			dev_hidden_dW,
			dev_hidden_db,
			dev_output_W,
			dev_output_b,
			dev_output_dW,
			dev_output_db,
			streamId
		);
}



// end of file
