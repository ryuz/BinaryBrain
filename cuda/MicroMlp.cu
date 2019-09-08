#include <iostream>
#include <algorithm>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bbcu/bbcu.h"
#include "bbcu/bbcu_util.h"
#include "Common.cuh"


// -------------------------------------------------
//  Forward
// -------------------------------------------------


template <int N=6, int M=16, int MAX_NODE_UNIT>
__global__ void kernal_fp32_MicroMlp_Forward
        (
            float const *x_buf,
            float       *y_buf,
            int   const *input_index,
            float const *hidden_W,
            float const *hidden_b,
            float const *output_W,
            float const *output_b,
            int         node_size,
            int         frame_size,
            int         frame_stride
        )
{
    int const node_id = threadIdx.y;
    int const node    = blockIdx.y * blockDim.y + threadIdx.y;
    int const id      = threadIdx.x;
    int const id_step = blockDim.x;

    // 係数読み込み
    __shared__ float        W0[M][N][MAX_NODE_UNIT];
    __shared__ float        b0[M][MAX_NODE_UNIT];
    __shared__ float        W1[M][MAX_NODE_UNIT];
    __shared__ float        b1[MAX_NODE_UNIT];
    __shared__ float const  *x_ptr[N][MAX_NODE_UNIT];
               float        *y_ptr;

    if ( node < node_size ) {
        for ( int i = id; i < M; i += id_step ) {
            for ( int j = 0; j < N; ++j ) {
                W0[i][j][node_id] = hidden_W[(node * M + i) * N + j];
            }

            b0[i][node_id] = hidden_b[node * M + i];
            W1[i][node_id] = output_W[node * M + i];
        }
        if ( id == 0 ) {
            b1[node_id] = output_b[node];
        }

        // 読み込みアドレス
        for ( int i = 0; i < N; ++i ) {
            int in_idx = input_index[node*N + i];
            x_ptr[i][node_id] = &x_buf[frame_stride * in_idx];
        }

        // 書き込みアドレス
        y_ptr = &y_buf[frame_stride * node];
    }
    
    __syncthreads();
    
    // 1つのSMで1nodeを全フレーム処理
    for ( int frame = id; frame < frame_size; frame += id_step ) {
        if ( node < node_size ) {
            // 入力データ読み込み
            float   x[N];
            for ( int i = 0; i < N; ++i ) {
                x[i] = x_ptr[i][node_id][frame];
            }

            // 計算
            float sig1 = b1[node_id];
            for ( int i = 0; i < M; ++i ) {
                float sig0 = b0[i][node_id];
                for ( int j = 0; j < N; ++j ) {
                    sig0 += x[j] * W0[i][j][node_id];
                }
        
                sig0 = fmaxf(sig0, 0);  // ReLU
        
                sig1 += sig0 * W1[i][node_id];
            }

            // 出力
            y_ptr[frame] = sig1;
        }
        __syncthreads();
    }
}

template <int N=6, int M=16>
int bbcu_fp32_MicroMlp_Forward
        (
            float const     *dev_x_buf,
            float           *dev_y_buf,
            int   const     *dev_input_index,
            float const     *dev_hidden_W,
            float const     *dev_hidden_b,
            float const     *dev_output_W,
            float const     *dev_output_b,
            int             input_node_size,
            int             output_node_size,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId = 0
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    unsigned int const THREAD_SIZE    = 512;
    unsigned int const MAX_FRAME_UNIT = 256;
    unsigned int const MAX_NODE_UNIT  = 16;

#if 0
    dim3    block(MAX_FRAME_UNIT, THREAD_SIZE / MAX_FRAME_UNIT);
    while ( (int)block.x / 2 >= frame_size )       { block.x /= 2; block.y *= 2; }
    while ( (int)block.y / 2 >= output_node_size ) { block.y /= 2; }
#else
    dim3    block(THREAD_SIZE / MAX_NODE_UNIT, MAX_NODE_UNIT);
    while ( (int)block.y / 2 >= output_node_size) { block.y /= 2; block.x *= 2;}
    while ( (int)block.x / 2 >= frame_size      ) { block.x /= 2; }
#endif

    block.x = std::min(block.x, MAX_FRAME_UNIT);
    block.y = std::min(block.y, MAX_NODE_UNIT);
    dim3    grid(1, (output_node_size + (block.y - 1)) / block.y);

    kernal_fp32_MicroMlp_Forward<N, M, MAX_NODE_UNIT><<<grid, block, 0, streamId>>>(
            dev_x_buf,
            dev_y_buf,
            dev_input_index,
            dev_hidden_W,
            dev_hidden_b,
            dev_output_W,
            dev_output_b,
            output_node_size,
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
            int             input_node_size,
            int             output_node_size,
            int             frame_size,
            int             frame_stride,
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



/////////////////////


// bit入力版
template <int N=6, int M=16, int MAX_NODE_UNIT=16>
__global__ void kernal_bit_fp32_MicroMlp_Forward(
            int const       *x_buf,
            float           *y_buf,
            int   const     *input_index,
            float const     *hidden_W,
            float const     *hidden_b,
            float const     *output_W,
            float const     *output_b,
            int             node_size,
            int             frame_size,
            int             input_frame_stride,
            int             output_frame_stride
        )
{
    int const node_id = threadIdx.y;
    int const node    = blockIdx.y * blockDim.y + threadIdx.y;
    int const id      = threadIdx.x;
    int const id_step = blockDim.x;

    // 係数読み込み
    __shared__ float        W0[M][N][MAX_NODE_UNIT];
    __shared__ float        b0[M][MAX_NODE_UNIT];
    __shared__ float        W1[M][MAX_NODE_UNIT];
    __shared__ float        b1[MAX_NODE_UNIT];
    __shared__ int const    *x_ptr[N][MAX_NODE_UNIT];

    if ( node < node_size) {
        for ( int i = id; i < M; i += id_step ) {
            for ( int j = 0; j < N; ++j ) {
                W0[i][j][node_id] = hidden_W[(node * M + i) * N + j];
            }

            b0[i][node_id] = hidden_b[node * M + i];
            W1[i][node_id] = output_W[node * M + i];
        }
        if ( id == 0 ) {
            b1[node_id] = output_b[node];
        }

        // 読み込みアドレス
        for ( int i = 0; i < N; ++i ) {
            int input_node = input_index[node*N + i];
            x_ptr[i][node_id] = &x_buf[input_frame_stride * input_node];
        }
    }
    
    __syncthreads();

    if ( node < node_size) {
        // 書き込みアドレス
        float *y_ptr = &y_buf[output_frame_stride * node];
    
        // 1つのSMで1nodeを全フレーム処理
        for ( int frame = id; frame < frame_size; frame += id_step ) {
            int bit  = (1 << (frame & 0x1f));
            int unit = (frame >> 5);

            // 入力データ読み込み
            int   x[N];
            for ( int i = 0; i < N; ++i ) {
                x[i] = x_ptr[i][node_id][unit];
            }
        
            // 計算
            float sig1 = b1[node_id];
            for ( int i = 0; i < M; ++i ) {
                float sig0 = b0[i][node_id];
                for ( int j = 0; j < N; ++j ) {
                    if ( x[j] & bit ) {
                        sig0 += W0[i][j][node_id];
                    }
                }
            
                sig0 = fmaxf(sig0, 0);  // ReLU
            
                sig1 += sig0 * W1[i][node_id];
            }

            // 出力
            y_ptr[frame] = sig1;
        }
    }
}


template <int N=6, int M=16>
int bbcu_bit_fp32_MicroMlp_Forward
        (
            int   const     *dev_x_buf,
            float           *dev_y_buf,
            int   const     *dev_input_index,
            float const     *dev_hidden_W,
            float const     *dev_hidden_b,
            float const     *dev_output_W,
            float const     *dev_output_b,
            int             input_node_size,
            int             output_node_size,
            int             frame_size,
            int             input_frame_stride,
            int             output_frame_stride,
            cudaStream_t    streamId = 0
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    unsigned int const THREAD_SIZE    = 512;
    unsigned int const MAX_FRAME_UNIT = 256;
    unsigned int const MAX_NODE_UNIT  = 16;

#if 0
    dim3    block(MAX_FRAME_UNIT, THREAD_SIZE / MAX_FRAME_UNIT);
    while ( (int)block.x / 2 >= frame_size )       { block.x /= 2; block.y *= 2; }
    while ( (int)block.y / 2 >= output_node_size ) { block.y /= 2; }
#else
    dim3    block(THREAD_SIZE / MAX_NODE_UNIT, MAX_NODE_UNIT);
    while ( (int)block.y / 2 >= output_node_size) { block.y /= 2; block.x *= 2;}
    while ( (int)block.x / 2 >= frame_size      ) { block.x /= 2; }
#endif

    block.x = std::min(block.x, MAX_FRAME_UNIT);
    block.y = std::min(block.y, MAX_NODE_UNIT);
    dim3    grid(1, (output_node_size + (block.y - 1)) / block.y);

    kernal_bit_fp32_MicroMlp_Forward<N, M, MAX_NODE_UNIT><<<grid, block, 0, streamId>>>
        (
            dev_x_buf,
            dev_y_buf,
            dev_input_index,
            dev_hidden_W,
            dev_hidden_b,
            dev_output_W,
            dev_output_b,
            output_node_size,
            frame_size,
            input_frame_stride,
            output_frame_stride
        );
    BB_CUDA_CHECK_LAST_ERROR();
    
    return 0;
}


int bbcu_bit_fp32_MicroMlp6x16_Forward
        (
            int   const     *dev_x_buf,
            float           *dev_y_buf,
            int   const     *dev_input_index,
            float const     *dev_hidden_W,
            float const     *dev_hidden_b,
            float const     *dev_output_W,
            float const     *dev_output_b,
            int             input_node_size,
            int             output_node_size,
            int             frame_size,
            int             input_frame_stride,
            int             output_frame_stride,
            cudaStream_t    streamId
        )
{
    return bbcu_bit_fp32_MicroMlp_Forward<6, 16>
        (
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
            input_frame_stride,
            output_frame_stride,
            streamId
        );
}




// -------------------------------------------------
//  Backward
// -------------------------------------------------

#if 0
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
#endif


// kernel
template <int N=6, int M=16, int MAX_FRAME_UNIT=32, int MAX_NODE_UNIT=8>
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
            int             node_size,
            int             frame_size,
            int             frame_stride
        )
{
    int const node_id = threadIdx.y;
    int const node    = blockIdx.y * blockDim.y + threadIdx.y;
    int const id      = threadIdx.x;
    int const id_step = blockDim.x;

    __shared__  float       sbuf[MAX_NODE_UNIT][MAX_FRAME_UNIT];
    __shared__  float       W0[M][N][MAX_NODE_UNIT];
    __shared__  float       b0[M][MAX_NODE_UNIT];
    __shared__  float       W1[M][MAX_NODE_UNIT];

    __shared__  float       dW0_prev[M][N][MAX_NODE_UNIT];
    __shared__  float       db0_prev[M][MAX_NODE_UNIT];
    __shared__  float       dW1_prev[M][MAX_NODE_UNIT];
    __shared__  float       db1_prev[MAX_NODE_UNIT];

    __shared__  float const *x_ptr[N][MAX_NODE_UNIT];

                float const *dy_ptr;

    if ( node < node_size ) {
        // 係数読み込み
        for ( int i = id; i < M; i += id_step ) {
            for ( int j = 0; j < N; ++j ) {
                W0[i][j][node_id] = hidden_W[(node * M + i) * N + j];
            }

            b0[i][node_id] = hidden_b[node * M + i];
            W1[i][node_id] = output_W[node * M + i];
        }

        // 直前の係数読み込み
        for ( int i = id; i < M; i += id_step ) {
            for ( int j = 0; j < N; ++j ) {
                dW0_prev[i][j][node_id] = hidden_dW[(node * M + i) * N + j];
            }

            db0_prev[i][node_id] = hidden_db[node * M + i];
            dW1_prev[i][node_id] = output_dW[node * M + i];
        }
        if ( id == 0 ) {
            db1_prev[node_id] = output_db[node];
        }

        // ポインタ読み込み
        for ( int i = 0; i < N; ++i ) {
            int input_node = input_index[node*N + i];
            x_ptr[i][node_id] = &x_buf[frame_stride * input_node];
        }

        dy_ptr = &dy_buf[frame_stride * node];
    }

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
    
    if ( node < node_size ) {
        // 1つのSMで1nodeを全フレーム処理
        for ( int frame = id; frame < frame_size; frame += id_step ) {
            // 入力データ読み込み
            float   x[N];
            for ( int i = 0; i < N; ++i ) {
                x[i] = x_ptr[i][node_id][frame];
            }
        
            // 1段目再計算して2段目逆伝播
            float   grad1 = dy_ptr[frame];
            float   grad0[M];
            db1 += grad1;
            for ( int i = 0; i < M; ++i ) {
                float sig0 = b0[i][node_id];
                for ( int j = 0; j < N; ++j ) {
                    sig0 += x[j] * W0[i][j][node_id];
                }
            
                sig0 = fmaxf(sig0, 0);  // ReLU

                dW1[i] += grad1 * sig0;

                if ( sig0 > 0 ) {       // ReLU
                    grad0[i] = grad1 * W1[i][node_id];
                }
                else {
                    grad0[i] = 0;
                }
            }
        
            // 1段目逆伝播
            float *dx_ptr  = &dx_buf[frame_stride * N * node];
            float   dx[N];
            for ( int i = 0; i < N; ++i ) {
                dx[i] = 0;  // dx_ptr[frame_stride * i + frame];
            }

            for ( int i = 0; i < M; ++i ) {
                db0[i] += grad0[i];
                for ( int j = 0; j < N; ++j ) {
                    dW0[i][j] += grad0[i] * x[j];
                    dx[j]     += grad0[i] * W0[i][j][node_id];
                }
            }
            
            // 誤差書き込み
            for ( int i = 0; i < N; ++i ) {
                dx_ptr[frame_stride * i + frame] = dx[i];
            }
        }
    }
    
    __syncthreads();

    // 係数統合
    for ( int i = 0; i < M; ++i ) {
        for ( int j = 0; j < N; ++j ) {
            dW0[i][j] = device_fp32_LocalSum(dW0[i][j], sbuf[node_id]);
        }
        db0[i] = device_fp32_LocalSum(db0[i], sbuf[node_id]);
        dW1[i] = device_fp32_LocalSum(dW1[i], sbuf[node_id]);
    }
    db1 = device_fp32_LocalSum(db1, sbuf[node_id]);

    // 勾配出力
    if ( node < node_size ) {
        for ( int i = id; i < M; i += id_step ) {
            for ( int j = 0; j < N; ++j ) {
                hidden_dW[(node * M + i) * N + j] = dW0[i][j] + dW0_prev[i][j][node_id];
            }
             hidden_db[node * M + i] = db0[i] + db0_prev[i][node_id];
             output_dW[node * M + i] = dW1[i] + dW1_prev[i][node_id];
        }
        if (id == 0) {
             output_db[node] = db1 + db1_prev[node_id];
        }
    }

    __syncthreads();
}


template <int N=6>
__global__ void kernal_fp32_MicroMlp_BackwardMarge
        (
            float const *src_buf,
            float       *dst_buf,
            int   const *input_index,
            int         node_size,
            int         frame_size,
            int         frame_stride
        )
{
    int frame = blockDim.x * blockIdx.x + threadIdx.x;
    
    for ( int node = 0; node < node_size; ++node ) {
        if ( frame < frame_size ) {
            for ( int n = 0; n < N; ++n ) {
                int in_idx = input_index[node*N + n];
                float*       dst_buf_ptr = &dst_buf[frame_stride * in_idx];
                float        prev_data   = dst_buf_ptr[frame];
                const float* src_buf_ptr = &src_buf[(N * node + n) * frame_stride];
                
                dst_buf_ptr[frame] = prev_data + src_buf_ptr[frame];
            }
        }
        __syncthreads();
    }
}



template <int N=6, int M=16>
int bbcu_fp32_MicroMlp_Backward
        (
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
            int             input_node_size,
            int             output_node_size,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId = 0
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    {
        unsigned int const THREAD_SIZE    = 256;
        unsigned int const MAX_FRAME_UNIT = 256;
        unsigned int const MAX_NODE_UNIT  = 16;

#if 0
        dim3    block(MAX_FRAME_UNIT, THREAD_SIZE / MAX_FRAME_UNIT);
        while ( (int)block.x / 2 >= frame_size )       { block.x /= 2; block.y *= 2; }
        while ( (int)block.y / 2 >= output_node_size ) { block.y /= 2; }
#else
        dim3    block(THREAD_SIZE / MAX_NODE_UNIT, MAX_NODE_UNIT);
        while ( (int)block.y / 2 >= output_node_size) { block.y /= 2; block.x *= 2;}
        while ( (int)block.x / 2 >= frame_size      ) { block.x /= 2; }
#endif

        block.x = std::min(block.x, MAX_FRAME_UNIT);
        block.y = std::min(block.y, MAX_NODE_UNIT);
        dim3    grid(1, (output_node_size + (block.y - 1)) / block.y);

        kernal_fp32_MicroMlp_Backward<N, M, MAX_FRAME_UNIT, MAX_NODE_UNIT><<<grid, block, 0, streamId>>>
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
                output_node_size,
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
            int             input_node_size,
            int             output_node_size,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId
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


///////////////////////////////


// kernel
template <int N=6, int M=16, int MAX_FRAME_UNIT=32, int MAX_NODE_UNIT=8>
__global__ void kernal_bit_fp32_MicroMlp_Backward
        (
            int   const     *x_buf,
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
            int             node_size,
            int             frame_size,
            int             x_frame_stride,
            int             frame_stride
        )
{
    int const node_id = threadIdx.y;
    int const node    = blockIdx.y * blockDim.y + threadIdx.y;
    int const id      = threadIdx.x;
    int const id_step = blockDim.x;

    __shared__  float       sbuf[MAX_NODE_UNIT][MAX_FRAME_UNIT];
    __shared__  float       W0[M][N][MAX_NODE_UNIT];
    __shared__  float       b0[M][MAX_NODE_UNIT];
    __shared__  float       W1[M][MAX_NODE_UNIT];

    __shared__  float       dW0_prev[M][N][MAX_NODE_UNIT];
    __shared__  float       db0_prev[M][MAX_NODE_UNIT];
    __shared__  float       dW1_prev[M][MAX_NODE_UNIT];
    __shared__  float       db1_prev[MAX_NODE_UNIT];

    __shared__  int   const *x_ptr[N][MAX_NODE_UNIT];

                float const *dy_ptr;

    if ( node < node_size ) {
        // 係数読み込み
        for ( int i = id; i < M; i += id_step ) {
            for ( int j = 0; j < N; ++j ) {
                W0[i][j][node_id] = hidden_W[(node * M + i) * N + j];
            }

            b0[i][node_id] = hidden_b[node * M + i];
            W1[i][node_id] = output_W[node * M + i];
        }

        // 直前の係数読み込み
        for ( int i = id; i < M; i += id_step ) {
            for ( int j = 0; j < N; ++j ) {
                dW0_prev[i][j][node_id] = hidden_dW[(node * M + i) * N + j];
            }

            db0_prev[i][node_id] = hidden_db[node * M + i];
            dW1_prev[i][node_id] = output_dW[node * M + i];
        }
        if ( id == 0 ) {
            db1_prev[node_id] = output_db[node];
        }

        // ポインタ読み込み
        for ( int i = 0; i < N; ++i ) {
            int input_node = input_index[node*N + i];
            x_ptr[i][node_id] = &x_buf[x_frame_stride * input_node];
        }

        dy_ptr = &dy_buf[frame_stride * node];
    }

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
    
    if ( node < node_size ) {
        // 1つのSMで1nodeを全フレーム処理
        for ( int frame = id; frame < frame_size; frame += id_step ) {
            int bit  = (1 << (frame & 0x1f));
            int unit = (frame >> 5);

            // 入力データ読み込み
            int   x[N];
            for ( int i = 0; i < N; ++i ) {
                x[i] = x_ptr[i][node_id][unit];
            }
            
            // 1段目再計算して2段目逆伝播
            float   grad1 = dy_ptr[frame];
            float   grad0[M];
            db1 += grad1;
            for ( int i = 0; i < M; ++i ) {
                float sig0 = b0[i][node_id];
                for ( int j = 0; j < N; ++j ) {
                    if ( x[j] & bit ) {
                        sig0 += W0[i][j][node_id];
                    }
                }
            
                sig0 = fmaxf(sig0, 0);  // ReLU

                dW1[i] += grad1 * sig0;

                if ( sig0 > 0 ) {       // ReLU
                    grad0[i] = grad1 * W1[i][node_id];
                }
                else {
                    grad0[i] = 0;
                }
            }
        
            // 1段目逆伝播
            float *dx_ptr  = &dx_buf[frame_stride * N * node];
            float   dx[N];
            for ( int i = 0; i < N; ++i ) {
                dx[i] = 0;  // dx_ptr[frame_stride * i + frame];
            }

            for ( int i = 0; i < M; ++i ) {
                db0[i] += grad0[i];
                for ( int j = 0; j < N; ++j ) {
                    if ( x[j] & bit ) { dW0[i][j] += grad0[i]; }
                    dx[j]     += grad0[i] * W0[i][j][node_id];
                }
            }
            
            // 誤差書き込み
            for ( int i = 0; i < N; ++i ) {
                dx_ptr[frame_stride * i + frame] = dx[i];
            }
        }
    }
    
    __syncthreads();

    // 係数統合
    for ( int i = 0; i < M; ++i ) {
        for ( int j = 0; j < N; ++j ) {
            dW0[i][j] = device_fp32_LocalSum(dW0[i][j], sbuf[node_id]);
        }
        db0[i] = device_fp32_LocalSum(db0[i], sbuf[node_id]);
        dW1[i] = device_fp32_LocalSum(dW1[i], sbuf[node_id]);
    }
    db1 = device_fp32_LocalSum(db1, sbuf[node_id]);

    // 勾配出力
    if ( node < node_size ) {
        for ( int i = id; i < M; i += id_step ) {
            for ( int j = 0; j < N; ++j ) {
                hidden_dW[(node * M + i) * N + j] = dW0[i][j] + dW0_prev[i][j][node_id];
            }
             hidden_db[node * M + i] = db0[i] + db0_prev[i][node_id];
             output_dW[node * M + i] = dW1[i] + dW1_prev[i][node_id];
        }
        if (id == 0) {
             output_db[node] = db1 + db1_prev[node_id];
        }
    }

    __syncthreads();
}


template <int N=6, int M=16>
int bbcu_bit_fp32_MicroMlp_Backward
        (
            int   const     *dev_x_buf,
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
            int             input_node_size,
            int             output_node_size,
            int             frame_size,
            int             x_frame_stride,
            int             frame_stride,
            cudaStream_t    streamId = 0
        )
{
    BBCU_DEBUG_ASSERT(bbcu_IsDeviceAvailable());

    {
        unsigned int const THREAD_SIZE    = 256;
        unsigned int const MAX_FRAME_UNIT = 256;
        unsigned int const MAX_NODE_UNIT  = 16;

#if 0
        dim3    block(MAX_FRAME_UNIT, THREAD_SIZE / MAX_FRAME_UNIT);
        while ( (int)block.x / 2 >= frame_size )       { block.x /= 2; block.y *= 2; }
        while ( (int)block.y / 2 >= output_node_size ) { block.y /= 2; }
#else
        dim3    block(THREAD_SIZE / MAX_NODE_UNIT, MAX_NODE_UNIT);
        while ( (int)block.y / 2 >= output_node_size) { block.y /= 2; block.x *= 2;}
        while ( (int)block.x / 2 >= frame_size      ) { block.x /= 2; }
#endif

        block.x = std::min(block.x, MAX_FRAME_UNIT);
        block.y = std::min(block.y, MAX_NODE_UNIT);
        dim3    grid(1, (output_node_size + (block.y - 1)) / block.y);

        kernal_bit_fp32_MicroMlp_Backward<N, M, MAX_FRAME_UNIT, MAX_NODE_UNIT><<<grid, block, 0, streamId>>>
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
                output_node_size,
                frame_size,
                x_frame_stride,
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


BBCU_DLL_EXPORT int bbcu_bit_fp32_MicroMlp6x16_Backward
        (
            int   const     *dev_x_buf,
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
            int             input_node_size,
            int             output_node_size,
            int             frame_size,
            int             x_frame_stride,
            int             frame_stride,
            cudaStream_t    streamId
        )
{
    return bbcu_bit_fp32_MicroMlp_Backward<6, 16>
        (
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
            x_frame_stride,
            frame_stride,
            streamId
        );
}



// end of file
