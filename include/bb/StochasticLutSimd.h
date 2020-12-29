// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once

#include <algorithm>
#include <array>
#include <vector>
#include "bb/SparseModel.h"
#include "bb/FrameBuffer.h"
#include "bb/FixedSizeConnectionTable.h"
#include "bb/Tensor.h"


namespace bb {


inline void simd_fp32_StochasticLut6_Forward
    (
        FrameBuffer                         x_buf,
        FrameBuffer                         y_buf,
        std::int32_t    const               *input_table,
        std::shared_ptr<Tensor>             W,
        bool                                binary_mode,
        bool                                lut_binarize,
        float                               unbinarize_bias
    )
{
    auto x_ptr           = x_buf.LockConst<float>();
    auto y_ptr           = y_buf.Lock<float>(true);
    auto W_ptr           = W->LockConst<float>();

    auto node_size  = y_buf.GetNodeSize();
    auto frame_size = y_buf.GetFrameStride() / (index_t)sizeof(float);

    #pragma omp parallel for
    for ( index_t node = 0; node < node_size; ++node ) {
        // read W
        __m256   W[64];
        for ( int i = 0; i < 64; ++i ) {
            float W_val = W_ptr(node, i);
            if ( lut_binarize ) {
                W_val = ((W_val > 0.5f) ? 1.0f : 0.0f);
            }
            W[i] = _mm256_set1_ps(W_val);
        }
        
        // read input index
        float const  *x_addr[6];
        for ( int i = 0; i < 6; ++i ) {
            x_addr[i] = x_ptr.GetAddr(input_table[node*6 + i]);
        }
        float *y_addr = y_ptr.GetAddr(node);

        for ( index_t frame = 0; frame < frame_size; frame += 8) {
            __m256   xp[6], xn[6];
            for ( int i = 0; i < 6; ++i) {
                xp[i] = _mm256_loadu_ps(&x_addr[i][frame]);
               if ( binary_mode ) {
                    __m256 mask =  _mm256_cmp_ps(xp[i], _mm256_set1_ps(0.5f), _CMP_GT_OS);
                    xp[i] = _mm256_blendv_ps(_mm256_set1_ps(0.5f - unbinarize_bias), _mm256_set1_ps(0.5f + unbinarize_bias), mask);
                }
                else {
                    xp[i] = _mm256_min_ps(xp[i], _mm256_set1_ps(1.0f));
                    xp[i] = _mm256_max_ps(xp[i], _mm256_set1_ps(0.0f));
               }
                xn[i] = _mm256_sub_ps(_mm256_set1_ps(1.0f), xp[i]);
            }

            __m256 x0_00 = _mm256_mul_ps(xn[1], xn[0]);
            __m256 x0_01 = _mm256_mul_ps(xn[1], xp[0]);
            __m256 x0_10 = _mm256_mul_ps(xp[1], xn[0]);
            __m256 x0_11 = _mm256_mul_ps(xp[1], xp[0]);
            __m256 x1_00 = _mm256_mul_ps(xn[3], xn[2]);
            __m256 x1_01 = _mm256_mul_ps(xn[3], xp[2]);
            __m256 x1_10 = _mm256_mul_ps(xp[3], xn[2]);
            __m256 x1_11 = _mm256_mul_ps(xp[3], xp[2]);
            __m256 x2_00 = _mm256_mul_ps(xn[5], xn[4]);
            __m256 x2_01 = _mm256_mul_ps(xn[5], xp[4]);
            __m256 x2_10 = _mm256_mul_ps(xp[5], xn[4]);
            __m256 x2_11 = _mm256_mul_ps(xp[5], xp[4]);

            __m256  y;
            y =   _mm256_mul_ps(W[0 ], _mm256_mul_ps(x2_00, _mm256_mul_ps(x1_00, x0_00)));
            y = _mm256_fmadd_ps(W[1 ], _mm256_mul_ps(x2_00, _mm256_mul_ps(x1_00, x0_01)), y);
            y = _mm256_fmadd_ps(W[2 ], _mm256_mul_ps(x2_00, _mm256_mul_ps(x1_00, x0_10)), y);
            y = _mm256_fmadd_ps(W[3 ], _mm256_mul_ps(x2_00, _mm256_mul_ps(x1_00, x0_11)), y);
            y = _mm256_fmadd_ps(W[4 ], _mm256_mul_ps(x2_00, _mm256_mul_ps(x1_01, x0_00)), y);
            y = _mm256_fmadd_ps(W[5 ], _mm256_mul_ps(x2_00, _mm256_mul_ps(x1_01, x0_01)), y);
            y = _mm256_fmadd_ps(W[6 ], _mm256_mul_ps(x2_00, _mm256_mul_ps(x1_01, x0_10)), y);
            y = _mm256_fmadd_ps(W[7 ], _mm256_mul_ps(x2_00, _mm256_mul_ps(x1_01, x0_11)), y);
            y = _mm256_fmadd_ps(W[8 ], _mm256_mul_ps(x2_00, _mm256_mul_ps(x1_10, x0_00)), y);
            y = _mm256_fmadd_ps(W[9 ], _mm256_mul_ps(x2_00, _mm256_mul_ps(x1_10, x0_01)), y);
            y = _mm256_fmadd_ps(W[10], _mm256_mul_ps(x2_00, _mm256_mul_ps(x1_10, x0_10)), y);
            y = _mm256_fmadd_ps(W[11], _mm256_mul_ps(x2_00, _mm256_mul_ps(x1_10, x0_11)), y);
            y = _mm256_fmadd_ps(W[12], _mm256_mul_ps(x2_00, _mm256_mul_ps(x1_11, x0_00)), y);
            y = _mm256_fmadd_ps(W[13], _mm256_mul_ps(x2_00, _mm256_mul_ps(x1_11, x0_01)), y);
            y = _mm256_fmadd_ps(W[14], _mm256_mul_ps(x2_00, _mm256_mul_ps(x1_11, x0_10)), y);
            y = _mm256_fmadd_ps(W[15], _mm256_mul_ps(x2_00, _mm256_mul_ps(x1_11, x0_11)), y);
            y = _mm256_fmadd_ps(W[16], _mm256_mul_ps(x2_01, _mm256_mul_ps(x1_00, x0_00)), y);
            y = _mm256_fmadd_ps(W[17], _mm256_mul_ps(x2_01, _mm256_mul_ps(x1_00, x0_01)), y);
            y = _mm256_fmadd_ps(W[18], _mm256_mul_ps(x2_01, _mm256_mul_ps(x1_00, x0_10)), y);
            y = _mm256_fmadd_ps(W[19], _mm256_mul_ps(x2_01, _mm256_mul_ps(x1_00, x0_11)), y);
            y = _mm256_fmadd_ps(W[20], _mm256_mul_ps(x2_01, _mm256_mul_ps(x1_01, x0_00)), y);
            y = _mm256_fmadd_ps(W[21], _mm256_mul_ps(x2_01, _mm256_mul_ps(x1_01, x0_01)), y);
            y = _mm256_fmadd_ps(W[22], _mm256_mul_ps(x2_01, _mm256_mul_ps(x1_01, x0_10)), y);
            y = _mm256_fmadd_ps(W[23], _mm256_mul_ps(x2_01, _mm256_mul_ps(x1_01, x0_11)), y);
            y = _mm256_fmadd_ps(W[24], _mm256_mul_ps(x2_01, _mm256_mul_ps(x1_10, x0_00)), y);
            y = _mm256_fmadd_ps(W[25], _mm256_mul_ps(x2_01, _mm256_mul_ps(x1_10, x0_01)), y);
            y = _mm256_fmadd_ps(W[26], _mm256_mul_ps(x2_01, _mm256_mul_ps(x1_10, x0_10)), y);
            y = _mm256_fmadd_ps(W[27], _mm256_mul_ps(x2_01, _mm256_mul_ps(x1_10, x0_11)), y);
            y = _mm256_fmadd_ps(W[28], _mm256_mul_ps(x2_01, _mm256_mul_ps(x1_11, x0_00)), y);
            y = _mm256_fmadd_ps(W[29], _mm256_mul_ps(x2_01, _mm256_mul_ps(x1_11, x0_01)), y);
            y = _mm256_fmadd_ps(W[30], _mm256_mul_ps(x2_01, _mm256_mul_ps(x1_11, x0_10)), y);
            y = _mm256_fmadd_ps(W[31], _mm256_mul_ps(x2_01, _mm256_mul_ps(x1_11, x0_11)), y);
            y = _mm256_fmadd_ps(W[32], _mm256_mul_ps(x2_10, _mm256_mul_ps(x1_00, x0_00)), y);
            y = _mm256_fmadd_ps(W[33], _mm256_mul_ps(x2_10, _mm256_mul_ps(x1_00, x0_01)), y);
            y = _mm256_fmadd_ps(W[34], _mm256_mul_ps(x2_10, _mm256_mul_ps(x1_00, x0_10)), y);
            y = _mm256_fmadd_ps(W[35], _mm256_mul_ps(x2_10, _mm256_mul_ps(x1_00, x0_11)), y);
            y = _mm256_fmadd_ps(W[36], _mm256_mul_ps(x2_10, _mm256_mul_ps(x1_01, x0_00)), y);
            y = _mm256_fmadd_ps(W[37], _mm256_mul_ps(x2_10, _mm256_mul_ps(x1_01, x0_01)), y);
            y = _mm256_fmadd_ps(W[38], _mm256_mul_ps(x2_10, _mm256_mul_ps(x1_01, x0_10)), y);
            y = _mm256_fmadd_ps(W[39], _mm256_mul_ps(x2_10, _mm256_mul_ps(x1_01, x0_11)), y);
            y = _mm256_fmadd_ps(W[40], _mm256_mul_ps(x2_10, _mm256_mul_ps(x1_10, x0_00)), y);
            y = _mm256_fmadd_ps(W[41], _mm256_mul_ps(x2_10, _mm256_mul_ps(x1_10, x0_01)), y);
            y = _mm256_fmadd_ps(W[42], _mm256_mul_ps(x2_10, _mm256_mul_ps(x1_10, x0_10)), y);
            y = _mm256_fmadd_ps(W[43], _mm256_mul_ps(x2_10, _mm256_mul_ps(x1_10, x0_11)), y);
            y = _mm256_fmadd_ps(W[44], _mm256_mul_ps(x2_10, _mm256_mul_ps(x1_11, x0_00)), y);
            y = _mm256_fmadd_ps(W[45], _mm256_mul_ps(x2_10, _mm256_mul_ps(x1_11, x0_01)), y);
            y = _mm256_fmadd_ps(W[46], _mm256_mul_ps(x2_10, _mm256_mul_ps(x1_11, x0_10)), y);
            y = _mm256_fmadd_ps(W[47], _mm256_mul_ps(x2_10, _mm256_mul_ps(x1_11, x0_11)), y);
            y = _mm256_fmadd_ps(W[48], _mm256_mul_ps(x2_11, _mm256_mul_ps(x1_00, x0_00)), y);
            y = _mm256_fmadd_ps(W[49], _mm256_mul_ps(x2_11, _mm256_mul_ps(x1_00, x0_01)), y);
            y = _mm256_fmadd_ps(W[50], _mm256_mul_ps(x2_11, _mm256_mul_ps(x1_00, x0_10)), y);
            y = _mm256_fmadd_ps(W[51], _mm256_mul_ps(x2_11, _mm256_mul_ps(x1_00, x0_11)), y);
            y = _mm256_fmadd_ps(W[52], _mm256_mul_ps(x2_11, _mm256_mul_ps(x1_01, x0_00)), y);
            y = _mm256_fmadd_ps(W[53], _mm256_mul_ps(x2_11, _mm256_mul_ps(x1_01, x0_01)), y);
            y = _mm256_fmadd_ps(W[54], _mm256_mul_ps(x2_11, _mm256_mul_ps(x1_01, x0_10)), y);
            y = _mm256_fmadd_ps(W[55], _mm256_mul_ps(x2_11, _mm256_mul_ps(x1_01, x0_11)), y);
            y = _mm256_fmadd_ps(W[56], _mm256_mul_ps(x2_11, _mm256_mul_ps(x1_10, x0_00)), y);
            y = _mm256_fmadd_ps(W[57], _mm256_mul_ps(x2_11, _mm256_mul_ps(x1_10, x0_01)), y);
            y = _mm256_fmadd_ps(W[58], _mm256_mul_ps(x2_11, _mm256_mul_ps(x1_10, x0_10)), y);
            y = _mm256_fmadd_ps(W[59], _mm256_mul_ps(x2_11, _mm256_mul_ps(x1_10, x0_11)), y);
            y = _mm256_fmadd_ps(W[60], _mm256_mul_ps(x2_11, _mm256_mul_ps(x1_11, x0_00)), y);
            y = _mm256_fmadd_ps(W[61], _mm256_mul_ps(x2_11, _mm256_mul_ps(x1_11, x0_01)), y);
            y = _mm256_fmadd_ps(W[62], _mm256_mul_ps(x2_11, _mm256_mul_ps(x1_11, x0_10)), y);
            y = _mm256_fmadd_ps(W[63], _mm256_mul_ps(x2_11, _mm256_mul_ps(x1_11, x0_11)), y);

            // clamp
            y = _mm256_max_ps(y, _mm256_set1_ps(0.0f));
            y = _mm256_min_ps(y, _mm256_set1_ps(1.0f));

            _mm256_storeu_ps(&y_addr[frame], y);
        }
    }
}


inline void simd_fp32_StochasticLut6_Backward
    (
        FrameBuffer                 x_buf,
        FrameBuffer                 dy_buf,
        FrameBuffer                 dx_buf,
        std::int32_t    const       *input_table,
        std::shared_ptr<Tensor>     W,
        std::shared_ptr<Tensor>     dW,
        float                       unbinarize_bias,
        bool                        binary_mode,
        bool                        lut_binarize
    )
{
    dx_buf.FillZero();

//  index_t input_node_size  = x_buf.GetNodeSize();
    index_t output_node_size = dy_buf.GetNodeSize();
    index_t frame_size       = dy_buf.GetFrameStride() / sizeof(float);

    // 並列化用tmpバッファ確保
    FrameBuffer dx_tmp(dy_buf.GetFrameSize(), {output_node_size * 6}, BB_TYPE_FP32);

    auto x_ptr           = x_buf.LockConst<float>();
    auto dy_ptr          = dy_buf.LockConst<float>();
    auto dx_ptr          = dx_buf.Lock<float>(true);
    auto dx_tmp_ptr      = dx_tmp.Lock<float>();
    auto W_ptr           = W->LockConst<float>();
    auto dW_ptr          = dW->Lock<float>();

    #pragma omp parallel for
    for ( index_t node = 0; node < output_node_size; ++node ) {           // initialize dW
        __m256  dW[64];
        for ( int i = 0; i < 64; ++i) {
            dW[i] = _mm256_set1_ps(0.0f);
        }

        // read W
        __m256   W[64];
        for ( int i = 0; i < 64; ++i ) {
            float W_val = W_ptr(node, i);
            if ( lut_binarize ) {
                W_val = W_val > 0.5f ? 1.0f : 0.0f;
            }
            W[i] = _mm256_set1_ps(W_val);
        }
    
        // read input index
        float const  *x_addr[6];
        for ( int i = 0; i < 6; ++i ) {
            x_addr[i] = x_ptr.GetAddr(input_table[node*6+ i]);
        }
        
        float const *dy_addr = dy_ptr.GetAddr(node);

        float   *dx_addr[6];
        for ( int i = 0; i < 6; ++i ) {
            dx_addr[i] = dx_tmp_ptr.GetAddr(node*6 + i);
        }
        
        for ( index_t frame = 0; frame < frame_size; frame += 8 ) {
            __m256   xp[6], xn[6];
            for ( int i = 0; i < 6; ++i) {
                xp[i] = _mm256_loadu_ps(&x_addr[i][frame]);
                if ( binary_mode ) {
                    __m256 mask =  _mm256_cmp_ps(xp[i], _mm256_set1_ps(0.5f), _CMP_GT_OS);
                    xp[i] = _mm256_blendv_ps(_mm256_set1_ps(0.5f - unbinarize_bias), _mm256_set1_ps(0.5f + unbinarize_bias), mask);
                }
                else {
                    xp[i] = _mm256_min_ps(xp[i], _mm256_set1_ps(1.0));
                    xp[i] = _mm256_max_ps(xp[i], _mm256_set1_ps(0.0));
                }
                xn[i] = _mm256_sub_ps(_mm256_set1_ps(1.0f), xp[i]);
            }

            __m256 x0_00 = _mm256_mul_ps(xn[1], xn[0]);
            __m256 x0_01 = _mm256_mul_ps(xn[1], xp[0]);
            __m256 x0_10 = _mm256_mul_ps(xp[1], xn[0]);
            __m256 x0_11 = _mm256_mul_ps(xp[1], xp[0]);
            __m256 x1_00 = _mm256_mul_ps(xn[3], xn[2]);
            __m256 x1_01 = _mm256_mul_ps(xn[3], xp[2]);
            __m256 x1_10 = _mm256_mul_ps(xp[3], xn[2]);
            __m256 x1_11 = _mm256_mul_ps(xp[3], xp[2]);
            __m256 x2_00 = _mm256_mul_ps(xn[5], xn[4]);
            __m256 x2_01 = _mm256_mul_ps(xn[5], xp[4]);
            __m256 x2_10 = _mm256_mul_ps(xp[5], xn[4]);
            __m256 x2_11 = _mm256_mul_ps(xp[5], xp[4]);

            __m256 grad = _mm256_load_ps(&dy_addr[frame]);
            dW[ 0] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_00, _mm256_mul_ps(x1_00, x0_00)), dW[ 0]);  
            dW[ 1] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_00, _mm256_mul_ps(x1_00, x0_01)), dW[ 1]);
            dW[ 2] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_00, _mm256_mul_ps(x1_00, x0_10)), dW[ 2]);
            dW[ 3] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_00, _mm256_mul_ps(x1_00, x0_11)), dW[ 3]);
            dW[ 4] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_00, _mm256_mul_ps(x1_01, x0_00)), dW[ 4]);
            dW[ 5] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_00, _mm256_mul_ps(x1_01, x0_01)), dW[ 5]);
            dW[ 6] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_00, _mm256_mul_ps(x1_01, x0_10)), dW[ 6]);
            dW[ 7] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_00, _mm256_mul_ps(x1_01, x0_11)), dW[ 7]);
            dW[ 8] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_00, _mm256_mul_ps(x1_10, x0_00)), dW[ 8]);
            dW[ 9] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_00, _mm256_mul_ps(x1_10, x0_01)), dW[ 9]);
            dW[10] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_00, _mm256_mul_ps(x1_10, x0_10)), dW[10]);
            dW[11] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_00, _mm256_mul_ps(x1_10, x0_11)), dW[11]);
            dW[12] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_00, _mm256_mul_ps(x1_11, x0_00)), dW[12]);
            dW[13] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_00, _mm256_mul_ps(x1_11, x0_01)), dW[13]);
            dW[14] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_00, _mm256_mul_ps(x1_11, x0_10)), dW[14]);
            dW[15] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_00, _mm256_mul_ps(x1_11, x0_11)), dW[15]);
            dW[16] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_01, _mm256_mul_ps(x1_00, x0_00)), dW[16]);
            dW[17] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_01, _mm256_mul_ps(x1_00, x0_01)), dW[17]);
            dW[18] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_01, _mm256_mul_ps(x1_00, x0_10)), dW[18]);
            dW[19] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_01, _mm256_mul_ps(x1_00, x0_11)), dW[19]);
            dW[20] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_01, _mm256_mul_ps(x1_01, x0_00)), dW[20]);
            dW[21] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_01, _mm256_mul_ps(x1_01, x0_01)), dW[21]);
            dW[22] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_01, _mm256_mul_ps(x1_01, x0_10)), dW[22]);
            dW[23] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_01, _mm256_mul_ps(x1_01, x0_11)), dW[23]);
            dW[24] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_01, _mm256_mul_ps(x1_10, x0_00)), dW[24]);
            dW[25] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_01, _mm256_mul_ps(x1_10, x0_01)), dW[25]);
            dW[26] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_01, _mm256_mul_ps(x1_10, x0_10)), dW[26]);
            dW[27] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_01, _mm256_mul_ps(x1_10, x0_11)), dW[27]);
            dW[28] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_01, _mm256_mul_ps(x1_11, x0_00)), dW[28]);
            dW[29] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_01, _mm256_mul_ps(x1_11, x0_01)), dW[29]);
            dW[30] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_01, _mm256_mul_ps(x1_11, x0_10)), dW[30]);
            dW[31] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_01, _mm256_mul_ps(x1_11, x0_11)), dW[31]);
            dW[32] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_10, _mm256_mul_ps(x1_00, x0_00)), dW[32]);
            dW[33] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_10, _mm256_mul_ps(x1_00, x0_01)), dW[33]);
            dW[34] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_10, _mm256_mul_ps(x1_00, x0_10)), dW[34]);
            dW[35] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_10, _mm256_mul_ps(x1_00, x0_11)), dW[35]);
            dW[36] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_10, _mm256_mul_ps(x1_01, x0_00)), dW[36]);
            dW[37] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_10, _mm256_mul_ps(x1_01, x0_01)), dW[37]);
            dW[38] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_10, _mm256_mul_ps(x1_01, x0_10)), dW[38]);
            dW[39] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_10, _mm256_mul_ps(x1_01, x0_11)), dW[39]);
            dW[40] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_10, _mm256_mul_ps(x1_10, x0_00)), dW[40]);
            dW[41] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_10, _mm256_mul_ps(x1_10, x0_01)), dW[41]);
            dW[42] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_10, _mm256_mul_ps(x1_10, x0_10)), dW[42]);
            dW[43] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_10, _mm256_mul_ps(x1_10, x0_11)), dW[43]);
            dW[44] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_10, _mm256_mul_ps(x1_11, x0_00)), dW[44]);
            dW[45] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_10, _mm256_mul_ps(x1_11, x0_01)), dW[45]);
            dW[46] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_10, _mm256_mul_ps(x1_11, x0_10)), dW[46]);
            dW[47] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_10, _mm256_mul_ps(x1_11, x0_11)), dW[47]);
            dW[48] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_11, _mm256_mul_ps(x1_00, x0_00)), dW[48]);
            dW[49] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_11, _mm256_mul_ps(x1_00, x0_01)), dW[49]);
            dW[50] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_11, _mm256_mul_ps(x1_00, x0_10)), dW[50]);
            dW[51] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_11, _mm256_mul_ps(x1_00, x0_11)), dW[51]);
            dW[52] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_11, _mm256_mul_ps(x1_01, x0_00)), dW[52]);
            dW[53] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_11, _mm256_mul_ps(x1_01, x0_01)), dW[53]);
            dW[54] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_11, _mm256_mul_ps(x1_01, x0_10)), dW[54]);
            dW[55] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_11, _mm256_mul_ps(x1_01, x0_11)), dW[55]);
            dW[56] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_11, _mm256_mul_ps(x1_10, x0_00)), dW[56]);
            dW[57] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_11, _mm256_mul_ps(x1_10, x0_01)), dW[57]);
            dW[58] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_11, _mm256_mul_ps(x1_10, x0_10)), dW[58]);
            dW[59] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_11, _mm256_mul_ps(x1_10, x0_11)), dW[59]);
            dW[60] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_11, _mm256_mul_ps(x1_11, x0_00)), dW[60]);
            dW[61] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_11, _mm256_mul_ps(x1_11, x0_01)), dW[61]);
            dW[62] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_11, _mm256_mul_ps(x1_11, x0_10)), dW[62]);
            dW[63] = _mm256_fmadd_ps(grad, _mm256_mul_ps(x2_11, _mm256_mul_ps(x1_11, x0_11)), dW[63]);

            __m256 dxi;
            __m256 dx0_00 = _mm256_set1_ps(0.0f);
            __m256 dx0_01 = _mm256_set1_ps(0.0f);
            __m256 dx0_10 = _mm256_set1_ps(0.0f);
            __m256 dx0_11 = _mm256_set1_ps(0.0f);
            __m256 dx1_00 = _mm256_set1_ps(0.0f);
            __m256 dx1_01 = _mm256_set1_ps(0.0f);
            __m256 dx1_10 = _mm256_set1_ps(0.0f);
            __m256 dx1_11 = _mm256_set1_ps(0.0f);
            __m256 dx2_00 = _mm256_set1_ps(0.0f);
            __m256 dx2_01 = _mm256_set1_ps(0.0f);
            __m256 dx2_10 = _mm256_set1_ps(0.0f);
            __m256 dx2_11 = _mm256_set1_ps(0.0f);
            dxi = _mm256_mul_ps(W[ 0], grad);  dx0_00 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_00, x1_00), dx0_00);  dx1_00 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_00, x0_00), dx1_00);  dx2_00 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_00, x0_00), dx2_00);
            dxi = _mm256_mul_ps(W[ 1], grad);  dx0_01 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_00, x1_00), dx0_01);  dx1_00 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_00, x0_01), dx1_00);  dx2_00 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_00, x0_01), dx2_00);
            dxi = _mm256_mul_ps(W[ 2], grad);  dx0_10 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_00, x1_00), dx0_10);  dx1_00 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_00, x0_10), dx1_00);  dx2_00 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_00, x0_10), dx2_00);
            dxi = _mm256_mul_ps(W[ 3], grad);  dx0_11 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_00, x1_00), dx0_11);  dx1_00 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_00, x0_11), dx1_00);  dx2_00 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_00, x0_11), dx2_00);
            dxi = _mm256_mul_ps(W[ 4], grad);  dx0_00 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_00, x1_01), dx0_00);  dx1_01 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_00, x0_00), dx1_01);  dx2_00 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_01, x0_00), dx2_00);
            dxi = _mm256_mul_ps(W[ 5], grad);  dx0_01 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_00, x1_01), dx0_01);  dx1_01 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_00, x0_01), dx1_01);  dx2_00 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_01, x0_01), dx2_00);
            dxi = _mm256_mul_ps(W[ 6], grad);  dx0_10 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_00, x1_01), dx0_10);  dx1_01 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_00, x0_10), dx1_01);  dx2_00 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_01, x0_10), dx2_00);
            dxi = _mm256_mul_ps(W[ 7], grad);  dx0_11 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_00, x1_01), dx0_11);  dx1_01 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_00, x0_11), dx1_01);  dx2_00 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_01, x0_11), dx2_00);
            dxi = _mm256_mul_ps(W[ 8], grad);  dx0_00 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_00, x1_10), dx0_00);  dx1_10 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_00, x0_00), dx1_10);  dx2_00 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_10, x0_00), dx2_00);
            dxi = _mm256_mul_ps(W[ 9], grad);  dx0_01 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_00, x1_10), dx0_01);  dx1_10 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_00, x0_01), dx1_10);  dx2_00 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_10, x0_01), dx2_00);
            dxi = _mm256_mul_ps(W[10], grad);  dx0_10 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_00, x1_10), dx0_10);  dx1_10 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_00, x0_10), dx1_10);  dx2_00 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_10, x0_10), dx2_00);
            dxi = _mm256_mul_ps(W[11], grad);  dx0_11 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_00, x1_10), dx0_11);  dx1_10 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_00, x0_11), dx1_10);  dx2_00 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_10, x0_11), dx2_00);
            dxi = _mm256_mul_ps(W[12], grad);  dx0_00 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_00, x1_11), dx0_00);  dx1_11 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_00, x0_00), dx1_11);  dx2_00 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_11, x0_00), dx2_00);
            dxi = _mm256_mul_ps(W[13], grad);  dx0_01 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_00, x1_11), dx0_01);  dx1_11 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_00, x0_01), dx1_11);  dx2_00 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_11, x0_01), dx2_00);
            dxi = _mm256_mul_ps(W[14], grad);  dx0_10 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_00, x1_11), dx0_10);  dx1_11 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_00, x0_10), dx1_11);  dx2_00 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_11, x0_10), dx2_00);
            dxi = _mm256_mul_ps(W[15], grad);  dx0_11 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_00, x1_11), dx0_11);  dx1_11 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_00, x0_11), dx1_11);  dx2_00 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_11, x0_11), dx2_00);
            dxi = _mm256_mul_ps(W[16], grad);  dx0_00 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_01, x1_00), dx0_00);  dx1_00 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_01, x0_00), dx1_00);  dx2_01 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_00, x0_00), dx2_01);
            dxi = _mm256_mul_ps(W[17], grad);  dx0_01 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_01, x1_00), dx0_01);  dx1_00 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_01, x0_01), dx1_00);  dx2_01 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_00, x0_01), dx2_01);
            dxi = _mm256_mul_ps(W[18], grad);  dx0_10 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_01, x1_00), dx0_10);  dx1_00 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_01, x0_10), dx1_00);  dx2_01 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_00, x0_10), dx2_01);
            dxi = _mm256_mul_ps(W[19], grad);  dx0_11 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_01, x1_00), dx0_11);  dx1_00 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_01, x0_11), dx1_00);  dx2_01 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_00, x0_11), dx2_01);
            dxi = _mm256_mul_ps(W[20], grad);  dx0_00 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_01, x1_01), dx0_00);  dx1_01 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_01, x0_00), dx1_01);  dx2_01 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_01, x0_00), dx2_01);
            dxi = _mm256_mul_ps(W[21], grad);  dx0_01 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_01, x1_01), dx0_01);  dx1_01 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_01, x0_01), dx1_01);  dx2_01 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_01, x0_01), dx2_01);
            dxi = _mm256_mul_ps(W[22], grad);  dx0_10 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_01, x1_01), dx0_10);  dx1_01 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_01, x0_10), dx1_01);  dx2_01 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_01, x0_10), dx2_01);
            dxi = _mm256_mul_ps(W[23], grad);  dx0_11 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_01, x1_01), dx0_11);  dx1_01 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_01, x0_11), dx1_01);  dx2_01 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_01, x0_11), dx2_01);
            dxi = _mm256_mul_ps(W[24], grad);  dx0_00 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_01, x1_10), dx0_00);  dx1_10 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_01, x0_00), dx1_10);  dx2_01 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_10, x0_00), dx2_01);
            dxi = _mm256_mul_ps(W[25], grad);  dx0_01 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_01, x1_10), dx0_01);  dx1_10 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_01, x0_01), dx1_10);  dx2_01 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_10, x0_01), dx2_01);
            dxi = _mm256_mul_ps(W[26], grad);  dx0_10 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_01, x1_10), dx0_10);  dx1_10 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_01, x0_10), dx1_10);  dx2_01 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_10, x0_10), dx2_01);
            dxi = _mm256_mul_ps(W[27], grad);  dx0_11 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_01, x1_10), dx0_11);  dx1_10 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_01, x0_11), dx1_10);  dx2_01 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_10, x0_11), dx2_01);
            dxi = _mm256_mul_ps(W[28], grad);  dx0_00 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_01, x1_11), dx0_00);  dx1_11 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_01, x0_00), dx1_11);  dx2_01 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_11, x0_00), dx2_01);
            dxi = _mm256_mul_ps(W[29], grad);  dx0_01 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_01, x1_11), dx0_01);  dx1_11 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_01, x0_01), dx1_11);  dx2_01 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_11, x0_01), dx2_01);
            dxi = _mm256_mul_ps(W[30], grad);  dx0_10 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_01, x1_11), dx0_10);  dx1_11 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_01, x0_10), dx1_11);  dx2_01 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_11, x0_10), dx2_01);
            dxi = _mm256_mul_ps(W[31], grad);  dx0_11 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_01, x1_11), dx0_11);  dx1_11 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_01, x0_11), dx1_11);  dx2_01 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_11, x0_11), dx2_01);
            dxi = _mm256_mul_ps(W[32], grad);  dx0_00 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_10, x1_00), dx0_00);  dx1_00 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_10, x0_00), dx1_00);  dx2_10 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_00, x0_00), dx2_10);
            dxi = _mm256_mul_ps(W[33], grad);  dx0_01 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_10, x1_00), dx0_01);  dx1_00 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_10, x0_01), dx1_00);  dx2_10 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_00, x0_01), dx2_10);
            dxi = _mm256_mul_ps(W[34], grad);  dx0_10 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_10, x1_00), dx0_10);  dx1_00 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_10, x0_10), dx1_00);  dx2_10 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_00, x0_10), dx2_10);
            dxi = _mm256_mul_ps(W[35], grad);  dx0_11 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_10, x1_00), dx0_11);  dx1_00 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_10, x0_11), dx1_00);  dx2_10 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_00, x0_11), dx2_10);
            dxi = _mm256_mul_ps(W[36], grad);  dx0_00 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_10, x1_01), dx0_00);  dx1_01 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_10, x0_00), dx1_01);  dx2_10 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_01, x0_00), dx2_10);
            dxi = _mm256_mul_ps(W[37], grad);  dx0_01 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_10, x1_01), dx0_01);  dx1_01 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_10, x0_01), dx1_01);  dx2_10 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_01, x0_01), dx2_10);
            dxi = _mm256_mul_ps(W[38], grad);  dx0_10 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_10, x1_01), dx0_10);  dx1_01 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_10, x0_10), dx1_01);  dx2_10 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_01, x0_10), dx2_10);
            dxi = _mm256_mul_ps(W[39], grad);  dx0_11 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_10, x1_01), dx0_11);  dx1_01 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_10, x0_11), dx1_01);  dx2_10 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_01, x0_11), dx2_10);
            dxi = _mm256_mul_ps(W[40], grad);  dx0_00 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_10, x1_10), dx0_00);  dx1_10 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_10, x0_00), dx1_10);  dx2_10 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_10, x0_00), dx2_10);
            dxi = _mm256_mul_ps(W[41], grad);  dx0_01 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_10, x1_10), dx0_01);  dx1_10 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_10, x0_01), dx1_10);  dx2_10 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_10, x0_01), dx2_10);
            dxi = _mm256_mul_ps(W[42], grad);  dx0_10 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_10, x1_10), dx0_10);  dx1_10 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_10, x0_10), dx1_10);  dx2_10 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_10, x0_10), dx2_10);
            dxi = _mm256_mul_ps(W[43], grad);  dx0_11 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_10, x1_10), dx0_11);  dx1_10 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_10, x0_11), dx1_10);  dx2_10 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_10, x0_11), dx2_10);
            dxi = _mm256_mul_ps(W[44], grad);  dx0_00 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_10, x1_11), dx0_00);  dx1_11 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_10, x0_00), dx1_11);  dx2_10 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_11, x0_00), dx2_10);
            dxi = _mm256_mul_ps(W[45], grad);  dx0_01 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_10, x1_11), dx0_01);  dx1_11 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_10, x0_01), dx1_11);  dx2_10 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_11, x0_01), dx2_10);
            dxi = _mm256_mul_ps(W[46], grad);  dx0_10 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_10, x1_11), dx0_10);  dx1_11 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_10, x0_10), dx1_11);  dx2_10 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_11, x0_10), dx2_10);
            dxi = _mm256_mul_ps(W[47], grad);  dx0_11 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_10, x1_11), dx0_11);  dx1_11 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_10, x0_11), dx1_11);  dx2_10 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_11, x0_11), dx2_10);
            dxi = _mm256_mul_ps(W[48], grad);  dx0_00 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_11, x1_00), dx0_00);  dx1_00 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_11, x0_00), dx1_00);  dx2_11 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_00, x0_00), dx2_11);
            dxi = _mm256_mul_ps(W[49], grad);  dx0_01 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_11, x1_00), dx0_01);  dx1_00 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_11, x0_01), dx1_00);  dx2_11 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_00, x0_01), dx2_11);
            dxi = _mm256_mul_ps(W[50], grad);  dx0_10 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_11, x1_00), dx0_10);  dx1_00 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_11, x0_10), dx1_00);  dx2_11 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_00, x0_10), dx2_11);
            dxi = _mm256_mul_ps(W[51], grad);  dx0_11 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_11, x1_00), dx0_11);  dx1_00 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_11, x0_11), dx1_00);  dx2_11 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_00, x0_11), dx2_11);
            dxi = _mm256_mul_ps(W[52], grad);  dx0_00 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_11, x1_01), dx0_00);  dx1_01 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_11, x0_00), dx1_01);  dx2_11 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_01, x0_00), dx2_11);
            dxi = _mm256_mul_ps(W[53], grad);  dx0_01 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_11, x1_01), dx0_01);  dx1_01 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_11, x0_01), dx1_01);  dx2_11 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_01, x0_01), dx2_11);
            dxi = _mm256_mul_ps(W[54], grad);  dx0_10 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_11, x1_01), dx0_10);  dx1_01 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_11, x0_10), dx1_01);  dx2_11 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_01, x0_10), dx2_11);
            dxi = _mm256_mul_ps(W[55], grad);  dx0_11 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_11, x1_01), dx0_11);  dx1_01 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_11, x0_11), dx1_01);  dx2_11 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_01, x0_11), dx2_11);
            dxi = _mm256_mul_ps(W[56], grad);  dx0_00 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_11, x1_10), dx0_00);  dx1_10 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_11, x0_00), dx1_10);  dx2_11 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_10, x0_00), dx2_11);
            dxi = _mm256_mul_ps(W[57], grad);  dx0_01 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_11, x1_10), dx0_01);  dx1_10 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_11, x0_01), dx1_10);  dx2_11 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_10, x0_01), dx2_11);
            dxi = _mm256_mul_ps(W[58], grad);  dx0_10 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_11, x1_10), dx0_10);  dx1_10 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_11, x0_10), dx1_10);  dx2_11 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_10, x0_10), dx2_11);
            dxi = _mm256_mul_ps(W[59], grad);  dx0_11 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_11, x1_10), dx0_11);  dx1_10 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_11, x0_11), dx1_10);  dx2_11 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_10, x0_11), dx2_11);
            dxi = _mm256_mul_ps(W[60], grad);  dx0_00 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_11, x1_11), dx0_00);  dx1_11 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_11, x0_00), dx1_11);  dx2_11 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_11, x0_00), dx2_11);
            dxi = _mm256_mul_ps(W[61], grad);  dx0_01 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_11, x1_11), dx0_01);  dx1_11 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_11, x0_01), dx1_11);  dx2_11 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_11, x0_01), dx2_11);
            dxi = _mm256_mul_ps(W[62], grad);  dx0_10 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_11, x1_11), dx0_10);  dx1_11 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_11, x0_10), dx1_11);  dx2_11 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_11, x0_10), dx2_11);
            dxi = _mm256_mul_ps(W[63], grad);  dx0_11 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_11, x1_11), dx0_11);  dx1_11 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x2_11, x0_11), dx1_11);  dx2_11 = _mm256_fmadd_ps(dxi, _mm256_mul_ps(x1_11, x0_11), dx2_11);
            
            __m256  dxn;
            __m256  dxp;
            dxn = _mm256_mul_ps  (dx0_00, xn[1]);
            dxn = _mm256_fmadd_ps(dx0_10, xp[1], dxn);
            dxp = _mm256_mul_ps  (dx0_01, xn[1]);
            dxp = _mm256_fmadd_ps(dx0_11, xp[1], dxp);
            _mm256_store_ps(&dx_addr[0][frame], _mm256_sub_ps(dxp, dxn));

            dxn = _mm256_mul_ps  (dx0_00, xn[0]);
            dxn = _mm256_fmadd_ps(dx0_01, xp[0], dxn);
            dxp = _mm256_mul_ps  (dx0_10, xn[0]);
            dxp = _mm256_fmadd_ps(dx0_11, xp[0], dxp);
            _mm256_store_ps(&dx_addr[1][frame], _mm256_sub_ps(dxp, dxn));

            dxn = _mm256_mul_ps  (dx1_00, xn[3]);
            dxp = _mm256_mul_ps  (dx1_01, xn[3]);
            dxn = _mm256_fmadd_ps(dx1_10, xp[3], dxn);
            dxp = _mm256_fmadd_ps(dx1_11, xp[3], dxp);  
            _mm256_store_ps(&dx_addr[2][frame], _mm256_sub_ps(dxp, dxn));

            dxn = _mm256_mul_ps  (dx1_00, xn[2]);
            dxn = _mm256_fmadd_ps(dx1_01, xp[2], dxn);
            dxp = _mm256_mul_ps  (dx1_10, xn[2]);
            dxp = _mm256_fmadd_ps(dx1_11, xp[2], dxp);
            _mm256_store_ps(&dx_addr[3][frame], _mm256_sub_ps(dxp, dxn));

            dxn = _mm256_mul_ps  (dx2_00, xn[5]);     
            dxp = _mm256_mul_ps  (dx2_01, xn[5]);     
            dxn = _mm256_fmadd_ps(dx2_10, xp[5], dxn); 
            dxp = _mm256_fmadd_ps(dx2_11, xp[5], dxp); 
            _mm256_store_ps(&dx_addr[4][frame], _mm256_sub_ps(dxp, dxn));

            dxn = _mm256_mul_ps  (dx2_00, xn[4]);
            dxn = _mm256_fmadd_ps(dx2_01, xp[4], dxn);
            dxp = _mm256_mul_ps  (dx2_10, xn[4]);
            dxp = _mm256_fmadd_ps(dx2_11, xp[4], dxp);
            _mm256_store_ps(&dx_addr[5][frame], _mm256_sub_ps(dxp, dxn));
        }
        
        // dW水平加算
        for ( int i = 0; i < 64; ++i) {
            dW_ptr(node, i) += bb_mm256_cvtss_f32(bb_mm256_hsum_ps(dW[i]));
        }
    }

    #pragma omp parallel for
    for ( index_t frame = 0; frame < frame_size; frame += 8 ) {
        for ( index_t node = 0; node < output_node_size; ++node ) {
            for ( int i = 0; i < 6; ++i) {
                float       *dx_addr     = dx_ptr.GetAddr(input_table[node*6+i]);
                __m256 dx  = _mm256_load_ps(&dx_addr[frame]);
                float const *dx_tmp_addr = dx_tmp_ptr.GetAddr(node*6 + i);
                __m256 tmp = _mm256_load_ps(&dx_tmp_addr[frame]);
                dx = _mm256_add_ps(dx, tmp);
                _mm256_store_ps(&dx_addr[frame], dx);
            }
        }
    }
}



}
