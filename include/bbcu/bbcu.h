﻿
#pragma once

#include <assert.h>

#ifdef BB_WITH_CUDA
#include "cuda_runtime.h"
#endif

#include "bb/Assert.h"
#include "bb/DataType.h"


#ifdef BBCU_DLL

// 基本的にbuccのみでのDLL化はしない
#ifdef DLL_EXPORT
#define BBCU_DLL_EXPORT __declspec(dllexport) 
#else
#define BBCU_DLL_EXPORT __declspec(dllimport) 
#endif

#else

#define BBCU_DLL_EXPORT /**/

#endif


#if defined(__cplusplus) && defined(BBCU_DLL)
extern "C" {
#endif



// -------------------------------------
//  Assert
// -------------------------------------

#define BBCU_ASSERT(x)          do { BB_ASSERT(x); } while(0)
#define BBCU_DEBUG_ASSERT(x)    do { BB_DEBUG_ASSERT(x); } while(0)



// -------------------------------------
//  Maneger
// -------------------------------------

BBCU_DLL_EXPORT int  bbcu_GetDeviceCount(void); 
BBCU_DLL_EXPORT int  bbcu_GetDevice(void); 
BBCU_DLL_EXPORT void bbcu_SetDevice(int device);

BBCU_DLL_EXPORT void bbcu_SetHostOnly(bool hostOnly);
BBCU_DLL_EXPORT bool bbcu_IsHostOnly(void);
BBCU_DLL_EXPORT bool bbcu_IsDeviceAvailable(void);



// -------------------------------------
//  Local Heap
// -------------------------------------

BBCU_DLL_EXPORT void  *bbcu_LocalHeap_Malloc(size_t size);
BBCU_DLL_EXPORT void   bbcu_LocalHeap_Free(void* ptr);
BBCU_DLL_EXPORT size_t bbcu_LocalHeap_GetMaxAllocSize(void);
BBCU_DLL_EXPORT size_t bbcu_LocalHeap_GetAllocatedSize(void);
BBCU_DLL_EXPORT void   bbcu_LocalHeap_GarbageCollect(void);

#if defined(__cplusplus) && defined(BBCU_DLL)
}
#endif


#include "bbcu/bbcu_util.h"



#if defined(__cplusplus) && defined(BBCU_DLL)
extern "C" {
#endif



// -------------------------------------
//  Vector Operation
// -------------------------------------

// dst = a
BBCU_DLL_EXPORT int bbcu_fp32_Vector_set(
            float*          dev_dst,
            float           a,
            int             size,
            cudaStream_t    streamId = 0
        );

// dst = a * src0 + b * src1 + c
BBCU_DLL_EXPORT int bbcu_fp32_Vector_add_ex
        (
            float*          dev_dst,
            const float*    dev_src0,
            const float*    dev_src1,
            float           a,
            float           b,
            float           c,
            int             size,
            cudaStream_t    streamId = 0
        );
    
// dst = a * src0 * src1 + b
BBCU_DLL_EXPORT int bbcu_fp32_Vector_mul_ex
        (
            float*          dev_dst,
            const float*    dev_src0,
            const float*    dev_src1,
            float           a,
            float           b,
            int             size,
            cudaStream_t    streamId = 0
        );

// dst = (a * src0 + b) / (c * src1 + d)
BBCU_DLL_EXPORT int bbcu_fp32_Vector_div_ex(
            float           *dev_dst,
            float const     *dev_src0,
            float const     *dev_src1,
            float           a,
            float           b,
            float           c,
            float           d,
            int             size,
            cudaStream_t    streamId = 0
        );

// dst = sqrt(src)
BBCU_DLL_EXPORT int bbcu_fp32_Vector_sqrt(
            float           *dev_dst,
            float const     *dev_src,
            int             size,
            cudaStream_t    streamId = 0
        );


// dst = exp(src)
BBCU_DLL_EXPORT int bbcu_fp32_Vector_exp(
            float           *dev_dst,
            float const     *dev_src,
            int             size,
            cudaStream_t    streamId = 0
        );


//  min(ベクトル同士)
BBCU_DLL_EXPORT int bbcu_fp32_Vector_min(
            float           *dev_dst,
            float const     *dev_src0,
            float const     *dev_src1,
            int             size,
            cudaStream_t    streamId = 0
        );

// min(係数)
BBCU_DLL_EXPORT int bbcu_fp32_Vector_min_v(
            float           *dev_dst,
            float const     *dev_src,
            float           a,
            int             size,
            cudaStream_t    streamId = 0
        );


//  max(ベクトル同士)
BBCU_DLL_EXPORT int bbcu_fp32_Vector_max(
            float           *dev_dst,
            float const     *dev_src0,
            float const     *dev_src1,
            int             size,
            cudaStream_t    streamId = 0
        );

// max(係数)
BBCU_DLL_EXPORT int bbcu_fp32_Vector_max_v(
            float           *dev_dst,
            float const     *dev_src,
            float           a,
            int             size,
            cudaStream_t    streamId = 0
        );


// clamp
BBCU_DLL_EXPORT int bbcu_fp32_Vector_clamp(
            float           *dev_dst,
            float const     *dev_src,
            float           lo,
            float           hi,
            int             size,
            cudaStream_t    streamId = 0
        );


// -------------------------------------
//  Matrix
// -------------------------------------

// Horizontal Sum
BBCU_DLL_EXPORT int bbcu_fp32_MatrixColwiseSum
        (
            const float*    dev_src,
            float*          dev_dst,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId = 0
        );


// Horizontal MeanVar
BBCU_DLL_EXPORT int bbcu_fp32_MatrixColwiseMeanVar
        (
            const float*    dev_src,
            float*          dev_mean,
            float*          dev_variance,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId = 0
        );


BBCU_DLL_EXPORT int bbcu_fp32_MatrixRowwiseSetVector
        (
            const float*    dev_x_vec,
            float*          dev_y_mat,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId = 0
        );



// -------------------------------------
//  Convert Type
// -------------------------------------

template<typename T=float>
BBCU_DLL_EXPORT int bbcu_ConvBitToReal
        (
            int   const     *dev_x_buf,
            T               *dev_y_buf,
            T               value0,
            T               value1,
            int             node_size,
            int             frame_size,
            int             x_frame_stride,
            int             y_frame_stride,
            cudaStream_t    streamId=0
        );


// -------------------------------------
//  FrameBufferCopy
// -------------------------------------

BBCU_DLL_EXPORT int bbcu_int32_FrameBufferCopy
        (
            int             *dev_dst_buf,
            int const       *dev_src_buf,
            int             node_size,
            int             dst_node_offset,
            int             src_node_offset,
            int             frame_size,
            int             dst_frame_offset,
            int             src_frame_offset,
            int             dst_frame_stride,
            int             src_frame_stride,
            cudaStream_t    streamId=0
        );


// -------------------------------------
//  Binary LUT
// -------------------------------------

int bbcu_bit_BinatyLut6_Forward
        (
            int const       *dev_x_buf,
            int             *dev_y_buf,
            int const       *dev_input_index,
            int const       *dev_table,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId = 0
        );


// -------------------------------------
//  ShuffleModulation
// -------------------------------------

BBCU_DLL_EXPORT int bbcu_bit_ShuffleModulation_Forward
        (
            int const       *dev_x_buf,
            int             *dev_y_buf,
            int const       *dev_table,
            int             shuffle_size,
            int             lowering_size,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId = 0
        );



// -------------------------------------
//  AverageLut
// -------------------------------------

template <typename T>
BBCU_DLL_EXPORT int bbcu_AverageLut_Forward
        (
            T   const       *dev_x_buf,
            T               *dev_y_buf,
            int const       *dev_input_index,
            int             n,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            bool            binarize_input,
            bool            binarize_output,
            cudaStream_t    streamId=0
        );

BBCU_DLL_EXPORT int bbcu_bit_AverageLut_Forward
        (
            int const       *dev_x_buf,
            int             *dev_y_buf,
            int const       *dev_input_index,
            int             n,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId=0
        );

template <typename T>
BBCU_DLL_EXPORT int bbcu_AverageLut_Backward
        (
            T   const       *dev_dy_buf,
            T               *dev_dx_buf,
            int const       *dev_input_index,
            int             n,
            int             input_node_size,
            int             output_node_size,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId=0
        );


// -------------------------------------
//  MaxLut
// -------------------------------------


template <typename T>
BBCU_DLL_EXPORT int bbcu_MaxLut_Forward
        (
            T   const       *dev_x_buf,
            T               *dev_y_buf,
            int const       *dev_input_index,
            int             n,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            bool            binarize_input,
            bool            binarize_output,
            cudaStream_t    streamId=0
        );

BBCU_DLL_EXPORT int bbcu_bit_MaxLut_Forward
        (
            int const       *dev_x_buf,
            int             *dev_y_buf,
            int const       *dev_input_index,
            int             n,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId=0
        );

template <typename T>
BBCU_DLL_EXPORT int bbcu_MaxLut_Backward
        (
            T   const       *dev_x_buf,
            T   const       *dev_dy_buf,
            T               *dev_dx_buf,
            int const       *dev_input_index,
            int             n,
            int             input_node_size,
            int             output_node_size,
            int             frame_size,
            int             frame_stride,
            bool            binarize_input,
            cudaStream_t    streamId=0
        );


template <typename T>
BBCU_DLL_EXPORT int bbcu_bit_MaxLut_Backward
        (
            int const       *dev_x_buf,
            T   const       *dev_dy_buf,
            T               *dev_dx_buf,
            int const       *dev_input_index,
            int             n,
            int             input_node_size,
            int             output_node_size,
            int             frame_size,
            int             x_frame_stride,
            int             dy_frame_stride,
            cudaStream_t    streamId=0
        );



// -------------------------------------
//  MicroMlp
// -------------------------------------

BBCU_DLL_EXPORT int bbcu_fp32_MicroMlp6x16_Forward
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
        );


BBCU_DLL_EXPORT int bbcu_bit_fp32_MicroMlp6x16_Forward
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
        );


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
            cudaStream_t    streamId = 0
        );

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
            cudaStream_t    streamId = 0
        );


// -------------------------------------
//  DifferentiableLut
// -------------------------------------

template <int N=6>
BBCU_DLL_EXPORT int bbcu_fp32_DifferentiableLutN_ForwardTraining
        (
            float const     *dev_x_buf,
            float           *dev_y_buf,
            int   const     *dev_input_index,
            float const     *dev_W,
            float           *dev_mean_buf,
            float           *dev_rstd_buf,
            float           *dev_running_mean_buf,
            float           *dev_running_var_buf,
            float           gamma,
            float           beta,
            float           momentum,
            float           unbinarize_bias,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            int             lut_binarize,
            int             binary_mode,
            cudaStream_t    streamId = 0
        );

template <int N=6>
BBCU_DLL_EXPORT int bbcu_bit_fp32_DifferentiableLutN_ForwardTraining
        (
            int   const     *dev_x_buf,
            int             *dev_y_buf,
            int   const     *dev_input_index,
            float const     *dev_W,
            float           *dev_mean_buf,
            float           *dev_rstd_buf,
            float           *dev_running_mean_buf,
            float           *dev_running_var_buf,
            float           gamma,
            float           beta,
            float           momentum,
            float           unbinarize_bias,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            int             lut_binarize,
            cudaStream_t    streamId = 0
        );

template <int N=6>
BBCU_DLL_EXPORT int bbcu_fp32_DifferentiableLutN_ForwardInference
        (
            float const     *dev_x_buf,
            float           *dev_y_buf,
            int   const     *dev_input_index,
            float const     *dev_W,
            float const     *running_mean_buf,
            float const     *running_var_buf,
            float           gamma,
            float           beta,
            float           unbinarize_bias,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            int             lut_binarize,
            int             binary_mode,
            cudaStream_t    streamId = 0
        );

template <int N=6>
BBCU_DLL_EXPORT int bbcu_bit_fp32_DifferentiableLutN_ForwardInference
        (
            int   const     *dev_x_buf,
            int             *dev_y_buf,
            int   const     *dev_input_index,
            float const     *dev_W,
            float const     *running_mean_buf,
            float const     *running_var_buf,
            float           gamma,
            float           beta,
            float           unbinarize_bias,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            int             lut_binarize,
            cudaStream_t    streamId = 0
        );

template <int N=6>
BBCU_DLL_EXPORT int bbcu_fp32_DifferentiableLutN_Backward
        (
            float const     *dev_x_buf,
            float const     *dev_dy_buf,
            float           *dev_dx_buf,
            float           *dev_dx_tmp,
            int   const     *dev_input_index,
            int   const     *dev_reverse_index,
            float const     *dev_W,
            float           *dev_dW,
            float const     *dev_mean_buf,
            float const     *dev_rstd_buf,
            float           *dev_dmean_tmp,
            float           *dev_dvar_tmp,
            float           gamma,
            float           beta,
            float           unbinarize_bias,
            int             reverse_index_stride,
            int             input_node_size,
            int             output_node_size,
            int             frame_size,
            int             frame_stride,
            int             tmp_frame_size,
            int             tmp_frame_stride,
            int             lut_binarize,
            int             binary_mode,
            cudaStream_t    streamId = 0
        );

template <int N=6>
BBCU_DLL_EXPORT int bbcu_bit_fp32_DifferentiableLutN_Backward
        (
            int   const     *dev_x_buf,
            float const     *dev_dy_buf,
            float           *dev_dx_buf,
            float           *dev_dx_tmp,
            int   const     *dev_input_index,
            int   const     *dev_reverse_index,
            float const     *dev_W,
            float           *dev_dW,
            float const     *dev_mean_buf,
            float const     *dev_rstd_buf,
            float           *dev_dmean_tmp,
            float           *dev_dvar_tmp,
            float           gamma,
            float           beta,
            float           unbinarize_bias,
            int             reverse_index_stride,
            int             input_node_size,
            int             output_node_size,
            int             frame_size,
            int             frame_stride,
            int             x_frame_stride,
            int             tmp_frame_size,
            int             tmp_frame_stride,
            int             lut_binarize,
            cudaStream_t    streamId = 0
        );


// -------------------------------------
//  StochasticLut
// -------------------------------------

template <int N=6>
BBCU_DLL_EXPORT int bbcu_fp32_StochasticLut_Forward
        (
            const float     *dev_x_buf,
            float           *dev_y_buf,
            int   const     *dev_input_index,
            float const     *dev_W,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            int             input_binary,
            int             lut_binarize,
            float           unbinarize_bias,
            cudaStream_t    streamId = 0
        );

template <int N=6>
BBCU_DLL_EXPORT int bbcu_bit_fp32_StochasticLut_Forward
        (
            int   const     *dev_x_buf,
            float           *dev_y_buf,
            int   const     *dev_input_index,
            float const     *dev_W,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            int             bin_frame_stride,
            int             lut_binarize,
            float           unbinarize_bias,
            cudaStream_t    streamId = 0
        );

template <int N=6>
BBCU_DLL_EXPORT int bbcu_bit_bit_fp32_StochasticLut_Forward
        (
            int   const     *dev_x_buf,
            int             *dev_y_buf,
            int   const     *dev_input_index,
            float const     *dev_W,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            int             lut_binarize,
            float           unbinarize_bias,
            cudaStream_t    streamId = 0
        );


template <int N=6>
BBCU_DLL_EXPORT int bbcu_fp32_StochasticLut_Backward
        (
            float const     *dev_x_buf,
            float const     *dev_dy_buf,
            float           *dev_dx_buf,
            float           *dev_dx_tmp,
            int   const     *dev_input_index,
            int   const     *dev_reverse_index,
            float const     *dev_W,
            float           *dev_dW,
            int             reverse_index_stride,
            int             input_node_size,
            int             output_node_size,
            int             frame_size,
            int             frame_stride,
            int             tmp_frame_size,
            int             tmp_frame_stride,
            int             input_binary,
            int             lut_binarize,
            float           unbinarize_bias,
            cudaStream_t    streamId = 0
        );

template <int N=6>
BBCU_DLL_EXPORT int bbcu_bit_fp32_StochasticLut_Backward
        (
            int   const     *dev_x_buf,
            float const     *dev_dy_buf,
            float           *dev_dx_buf,
            float           *dev_dx_tmp,
            int   const     *dev_input_index,
            int   const     *dev_reverse_index,
            float const     *dev_W,
            float           *dev_dW,
            int             reverse_index_stride,
            int             input_node_size,
            int             output_node_size,
            int             frame_size,
            int             frame_stride,
            int             bin_frame_stride,
            int             tmp_frame_size,
            int             tmp_frame_stride,
            int             lut_binarize,
            float           unbinarize_bias,
            cudaStream_t    streamId = 0
        );


// -------------------------------------
//  StochasticMaxPooling
// -------------------------------------

BBCU_DLL_EXPORT int bbcu_fp32_StochasticMaxPooling2x2_Forward
        (
            float const *   dev_x_buf,
            float*          dev_y_buf,
            int             input_w_size,
            int             input_h_size,
            int             output_w_size,
            int             output_h_size,
            int             c_size,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId = 0
        );


BBCU_DLL_EXPORT int bbcu_fp32_StochasticMaxPooling2x2_Backward
        (
            float const     *dev_x_buf,
            float const     *dev_dy_buf,
            float           *dev_dx_buf,
            int             input_w_size,
            int             input_h_size,
            int             output_w_size,
            int             output_h_size,
            int             c_size,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId = 0
        );


// -------------------------------------
//  BatchNormalization
// -------------------------------------

BBCU_DLL_EXPORT int bbcu_fp32_BatchNormalization_ForwardTraining
        (
            float const     *dev_x_buf,
            float           *dev_y_buf,
            float const     *dev_gamma_buf,
            float const     *dev_beta_buf,
            float           *dev_mean_buf,
            float           *dev_rstd_buf,
            float           *dev_running_mean_buf,
            float           *dev_running_var_buf,
            float           momentum,
            int             node_size,  
            int             frame_size,
            int             frame_stride,
            bool            update_running_param,
            cudaStream_t    streamId=0
        );

BBCU_DLL_EXPORT int bbcu_fp32_BatchNormalization_ReForward
        (
            float const     *dev_x_buf,
            float           *dev_y_buf,
            float const     *dev_gamma_buf,
            float const     *dev_beta_buf,
            float const     *dev_mean_buf,
            float const     *dev_rstd_buf,
            int             node_size,  
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId = 0
        );

BBCU_DLL_EXPORT int bbcu_fp32_BatchNormalization_ForwardInference
        (
            float const     *dev_x_buf,
            float           *dev_y_buf,
            float const     *dev_gamma_buf,
            float const     *dev_beta_buf,
            float const     *dev_running_mean_buf,
            float const     *dev_running_var_buf,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId = 0
        );

BBCU_DLL_EXPORT int bbcu_fp32_BatchNormalization_Backward
        (
            float const     *dev_x_buf,
            float const     *dev_dy_buf,
            float           *dev_dx_buf,
            float const     *dev_gamma_buf,
            float           *dev_dgamma_buf,
            float           *dev_dbeta_buf,
            float const     *dev_mean_buf,
            float const     *dev_rstd_buf,
            float           reciprocal_frame_size,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId = 0
        );

BBCU_DLL_EXPORT int bbcu_fp32_BatchNormalization_BackwardLock
        (
            float const     *dev_dy_buf,
            float           *dev_dx_buf,
            float const     *dev_gamma_buf,
            float const     *dev_running_var_buf,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId = 0
        );


// -------------------------------------
//  Stochastic BatchNormalization
// -------------------------------------

BBCU_DLL_EXPORT int bbcu_fp32_StochasticBatchNormalization_ForwardTraining
        (
            float const     *dev_x_buf,
            float           *dev_y_buf,
            float           *dev_mean_buf,
            float           *dev_rstd_buf,
            float           *dev_running_mean_buf,
            float           *dev_running_var_buf,
            float           gamma,
            float           beta,
            float           momentum,
            int             node_size,  
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId = 0
        );

BBCU_DLL_EXPORT int bbcu_fp32_StochasticBatchNormalization_ReForward
        (
            float const     *dev_x_buf,
            float           *dev_y_buf,
            float const     *dev_mean_buf,
            float const     *dev_rstd_buf,
            float           gamma,
            float           beta,
            int             node_size,  
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId = 0
        );

BBCU_DLL_EXPORT int bbcu_fp32_StochasticBatchNormalization_ForwardInference
        (
            float const     *dev_x_buf,
            float           *dev_y_buf,
            float const     *dev_running_mean_buf,
            float const     *dev_running_var_buf,
            float           gamma,
            float           beta,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId = 0
        );

BBCU_DLL_EXPORT int bbcu_fp32_StochasticBatchNormalization_Backward
        (
            float const     *dev_x_buf,
            float const     *dev_dy_buf,
            float           *dev_dx_buf,
            float const     *dev_mean_buf,
            float const     *dev_rstd_buf,
            float           gamma,
            float           reciprocal_frame_size,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            int             x_frame_stride,
            cudaStream_t    streamId = 0
        );


// -------------------------------------
//  MaxPooling
// -------------------------------------

BBCU_DLL_EXPORT int bbcu_fp32_MaxPooling_Forward
        (
            float const *   dev_x_buf,
            float*          dev_y_buf,
            int*            dev_argmax_buf,
            int             filter_h_size,
            int             filter_w_size,
            int             input_w_size,
            int             input_h_size,
            int             output_w_size,
            int             output_h_size,
            int             c_size,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId = 0
        );

BBCU_DLL_EXPORT int bbcu_bit_MaxPooling_Forward
        (
            int const       *dev_x_buf,
            int             *dev_y_buf,
            int             filter_h_size,
            int             filter_w_size,
            int             input_w_size,
            int             input_h_size,
            int             output_w_size,
            int             output_h_size,
            int             c_size,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId = 0
        );


BBCU_DLL_EXPORT int bbcu_fp32_MaxPooling_Backward
        (
            int   const     *dev_argmax_buf,
            float const     *dev_dy_buf,
            float           *dev_dx_buf,
            int             filter_h_size,
            int             filter_w_size,
            int             input_w_size,
            int             input_h_size,
            int             output_w_size,
            int             output_h_size,
            int             c_size,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId = 0
        );

BBCU_DLL_EXPORT int bbcu_bit_fp32_MaxPooling_Backward
        (
            int   const     *dev_x_buf,
            float const     *dev_dy_buf,
            float           *dev_dx_buf,
            int             filter_h_size,
            int             filter_w_size,
            int             input_w_size,
            int             input_h_size,
            int             output_w_size,
            int             output_h_size,
            int             c_size,
            int             frame_size,
            int             forward_frame_stride,
            int             backward_frame_stride,
            cudaStream_t    streamId = 0
        );



// -------------------------------------
//  UpSampling
// -------------------------------------

BBCU_DLL_EXPORT int bbcu_fp32_UpSampling_Forward
        (
            float const     *dev_x_buf,
            float           *dev_y_buf,
            int             input_w_size,
            int             input_h_size,
            int             c_size,
            int             filter_w_size,
            int             filter_h_size,
            int             fill,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId=0
        );

BBCU_DLL_EXPORT int bbcu_bit_UpSampling_Forward
        (
            int const       *dev_x_buf,
            int             *dev_y_buf,
            int             input_w_size,
            int             input_h_size,
            int             c_size,
            int             filter_w_size,
            int             filter_h_size,
            int             fill,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId=0
        );


BBCU_DLL_EXPORT int bbcu_fp32_UpSampling_Backward
        (
            float const     *dev_dy_buf,
            float           *dev_dx_buf,
            int             input_w_size,
            int             input_h_size,
            int             c_size,
            int             filter_w_size,
            int             filter_h_size,
            int             fill,          
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId=0
        );



// -------------------------------------
//  Im2Col
// -------------------------------------

BBCU_DLL_EXPORT int bbcu_fp32_Im2Col_Forward
        (
            float const     *dev_x_buf,
            float           *dev_y_buf,
            int             x_stride,
            int             y_stride,
            int             x_offset,
            int             y_offset,
            int             input_frame_size,
            int             input_frame_stride,
            int             input_w_size,
            int             input_h_size,
            int             input_c_size,
            int             output_w_size,
            int             output_h_size,
            int             output_frame_stride,
            int             filter_w_size,
            int             filter_h_size,
            int             border_mode,
            float           border_value,
            cudaStream_t    streamId = 0
        );

BBCU_DLL_EXPORT int bbcu_bit_Im2Col_Forward
        (
            int const       *dev_x_buf,
            int             *dev_y_buf,
            int             x_stride,
            int             y_stride,
            int             x_offset,
            int             y_offset,
            int             input_frame_size,
            int             input_frame_stride,
            int             input_w_size,
            int             input_h_size,
            int             input_c_size,
            int             output_w_size,
            int             output_h_size,
            int             output_frame_stride,
            int             filter_w_size,
            int             filter_h_size,
            int             border_mode,
            cudaStream_t    streamId = 0
        );


BBCU_DLL_EXPORT int bbcu_fp32_Im2Col_Backward
        (
            float const     *dev_fy_buf,
            float           *dev_dx_buf,
            int             x_stride,
            int             y_stride,
            int             x_offset,
            int             y_offset,
            int             input_frame_size,
            int             input_frame_stride,
            int             input_w_size,
            int             input_h_size,
            int             input_c_size,
            int             output_w_size,
            int             output_h_size,
            int             output_frame_stride,            
            int             filter_w_size,
            int             filter_h_size,            
            cudaStream_t    streamId = 0
        );


// -------------------------------------
//  Col2Im
// -------------------------------------

BBCU_DLL_EXPORT int bbcu_fp32_Col2Im_Forward
        (
            float const     *dev_x_buf,
            float           *dev_y_buf,
            int             w_size,
            int             h_size,
            int             c_size,
            int             input_frame_stride,
            int             output_frame_size,
            int             output_frame_stride,
            cudaStream_t    streamId = 0
        );

BBCU_DLL_EXPORT int bbcu_bit_Col2Im_Forward
        (
            int const       *dev_x_buf,
            int             *dev_y_buf,
            int             w_size,
            int             h_size,
            int             c_size,
            int             input_frame_stride,
            int             output_frame_size,
            int             output_frame_stride,
            cudaStream_t    streamId = 0
        );


BBCU_DLL_EXPORT int bbcu_fp32_Col2Im_Backward
        (
            float const     *dev_dy_buf,
            float           *dev_dx_buf,
            int             w_size,
            int             h_size,
            int             c_size,
            int             input_frame_stride,
            int             output_frame_size,
            int             output_frame_stride,
            cudaStream_t    streamId = 0
        );



// -------------------------------------
//  Shuffle
// -------------------------------------

template<typename T>
BBCU_DLL_EXPORT int bbcu_Shuffle_Forward
        (
            T const         *dev_x_buf,
            T               *dev_y_buf,
            unsigned int    y_unit_size,
            unsigned int    node_size,
            unsigned int    frame_size,
            unsigned int    frame_stride,
            cudaStream_t    streamId = 0
        );

template<typename T>
BBCU_DLL_EXPORT int bbcu_Shuffle_Backward
        (
            T const         *dev_dy_buf,
            T               *dev_dx_buf,
            unsigned int    y_unit_size,
            unsigned int    node_size,
            unsigned int    frame_size,
            unsigned int    frame_stride,
            cudaStream_t    streamId = 0
        );


// -------------------------------------
//  Binarize
// -------------------------------------

BBCU_DLL_EXPORT int bbcu_fp32_Binarize_Forward
        (
            const float*    dev_x_buf,
            float*          dev_y_buf,
            float           binary_th,
            float           binary_low,
            float           binary_high,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId = 0
        );

BBCU_DLL_EXPORT int bbcu_fp32_bit_Binarize_Forward
        (
            float const     *dev_x_buf,
            int             *dev_y_buf,
            float           binary_th,
            int             node_size,
            int             frame_size,
            int             x_frame_stride,
            int             y_frame_stride,
            cudaStream_t    streamId = 0
        );


// -------------------------------------
//  Hard-Tanh
// -------------------------------------

BBCU_DLL_EXPORT int bbcu_fp32_HardTanh_Forward
        (
            float const *   dev_x_buf,
            float*          dev_y_buf,
            float           limit_min,
            float           limit_max,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId = 0
        );

BBCU_DLL_EXPORT int bbcu_fp32_HardTanh_Backward
        (
            const float*    dev_x_buf,
            const float*    dev_dy_buf,
            float*          dev_dx_buf,
            float           limit_min,
            float           limit_max,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId = 0
        );


// -------------------------------------
//  Sigmoid
// -------------------------------------

BBCU_DLL_EXPORT int bbcu_fp32_Sigmoid_Forward
        (
            const float*    dev_x_buf,
            float*          dev_y_buf,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId = 0
        );


BBCU_DLL_EXPORT int bbcu_fp32_Sigmoid_Backward
        (
            const float*    dev_y_buf,
            const float*    dev_dy_buf,
            float*          dev_dx_buf,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId = 0
        );


// -------------------------------------
//  ReLU
// -------------------------------------

BBCU_DLL_EXPORT int bbcu_fp32_ReLU_Forward
        (
            const float*    dev_x_buf,
            float*          dev_y_buf,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId = 0
        );

BBCU_DLL_EXPORT int bbcu_fp32_ReLU_Backward
        (
            const float*    dev_x_buf,
            const float*    dev_dy_buf,
            float*          dev_dx_buf,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId = 0
        );


// -------------------------------------
//  RealToBinary
// -------------------------------------

template<typename T=float>
BBCU_DLL_EXPORT int bbcu_RealToBinary_Forward
        (
            T   const       *dev_x_buf,
            T               *dev_y_buf,
            unsigned int    depth_modulation_size,
            unsigned int    frame_modulation_size,
            T               input_range_lo,
            T               input_range_hi,
            unsigned int    point_size,
            unsigned int    x_depth_size,
            unsigned int    x_frame_size,
            unsigned int    x_frame_stride,
            unsigned int    y_frame_stride,
            bool            binarize,
            cudaStream_t    streamId=0
        );

template<typename T=float>
BBCU_DLL_EXPORT int bbcu_bit_RealToBinary_Forward
        (
            T   const       *dev_x_buf,
            int             *dev_y_buf,
            unsigned int    depth_modulation_size,
            unsigned int    frame_modulation_size,
            T               input_range_lo,
            T               input_range_hi,
            unsigned int    point_size,
            unsigned int    x_depth_size,
            unsigned int    x_frame_size,
            unsigned int    x_frame_stride,
            unsigned int    y_frame_stride,
            cudaStream_t    streamId=0
        );


// -------------------------------------
//  BinaryToReal
// -------------------------------------

BBCU_DLL_EXPORT int bbcu_fp32_BinaryToReal_Forward
        (
            float const     *dev_x_buf,
            float           *dev_y_buf,
            int             node_mux_size,
            int             frame_mux_size,
            int             y_node_size,
            int             x_frame_stride,
            int             y_frame_size,
            int             y_frame_stride,
            cudaStream_t    streamId = 0
        );

BBCU_DLL_EXPORT int bbcu_bit_fp32_BinaryToReal_Forward
        (
            int   const     *dev_x_buf,
            float           *dev_y_buf,
            int             node_mux_size,
            int             frame_mux_size,
            int             y_node_size,
            int             x_frame_stride,
            int             y_frame_size,
            int             y_frame_stride,
            cudaStream_t    streamId = 0
        );

BBCU_DLL_EXPORT int bbcu_fp32_BinaryToReal_Backward
        (
            float  const    *dev_dy_buf,
            float           *dev_dx_buf,
            int             node_mux_size,
            int             frame_mux_size,
            int             y_node_size,
            int             x_frame_stride,
            int             y_frame_size,
            int             y_frame_stride,
            cudaStream_t    streamId = 0
        );



// -------------------------------------
//  BitEncode
// -------------------------------------

template<typename T>
BBCU_DLL_EXPORT int bbcu_bit_BitEncode
        (
            T const         *dev_x_buf,
            int             *dev_y_buf,
            unsigned int    bit_size,
            T               clip_min,
            T               clip_max,
            T               scale,
            T               offset,
            unsigned int    node_size,
            unsigned int    frame_size,
            unsigned int    x_frame_stride,
            unsigned int    y_frame_stride,
            cudaStream_t    streamId=0
        );


// -------------------------------------
//  BitError
// -------------------------------------


BBCU_DLL_EXPORT int bbcu_fp32_BitError_Forward
(
    float*              dev_x_buf,
    int                 seed,
    double              error_rate,
    bool                mask_mode,
    int                 node_size,
    int                 frame_size,
    int                 frame_stride,
    cudaStream_t        streamId=0
);

BBCU_DLL_EXPORT int bbcu_bit_BitError_Forward
(
    int*                dev_x_buf,
    int                 seed,
    double              error_rate,
    bool                mask_mode,
    int                 node_size,
    int                 frame_size,
    int                 frame_stride,
    cudaStream_t        streamId=0
);


BBCU_DLL_EXPORT int bbcu_fp32_BitError_Backward
(
    float*              dev_dy_buf,
    int                 seed,
    double              error_rate,
    float               weight0,
    float               weight1,
    int                 node_size,
    int                 frame_size,
    int                 frame_stride,
    cudaStream_t        streamId=0
);


// -------------------------------------
//  LossMeanSquaredError
// -------------------------------------

BBCU_DLL_EXPORT int bbcu_fp32_LossMeanSquaredError
        (
            const float*    dev_y_buf,
            const float*    dev_t_buf,
            float*          dev_dy_buf,
            double*         dev_loss_buf,
            int             loss_buf_size,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            float           grad_reduction,
            double          loss_reduction,
            cudaStream_t    streamId = 0
        );

// -------------------------------------
//  LossSoftmaxCrossEntropy
// -------------------------------------

template<typename T>
BBCU_DLL_EXPORT int bbcu_LossSoftmaxCrossEntropy
        (
            T   const       *dev_y_buf,
            T   const       *dev_t_buf,
            T               *dev_dy_buf,
            double          *dev_loss_buf,
            T               t_sum,
            int             pix_size,
            int             ch_size,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId=0
        );

BBCU_DLL_EXPORT int bbcu_fp32_LossSoftmaxCrossEntropy
        (
            float const     *dev_y_buf,
            float const     *dev_t_buf,
            float           *dev_dy_buf,
            double          *dev_loss_buf,
            double          *dev_loss,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            int             batch_size,
            cudaStream_t    streamId = 0
        );


// -------------------------------------
//  MetricsCategoricalAccuracy
// -------------------------------------

template<typename T>
BBCU_DLL_EXPORT int bbcu_MetricsCategoricalAccuracy
        (
            T   const       *dev_y_buf,
            T   const       *dev_t_buf,
            int             *dev_accuracy_buf,
            int             *dev_category_buf,
            int             pix_size,
            int             ch_size,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId=0
        );



// -------------------------------------
//  Adam
// -------------------------------------

BBCU_DLL_EXPORT int bbcu_fp32_OptimizerAdam
        (
            int             size,
            int     const   *dev_size_table,
            float * const   *dev_params_buf_table,
            float * const   *dev_grads_buf_table,
            float * const   *dev_m_buf_table,
            float * const   *dev_v_buf_table,
            float           lr_t,
            float           beta1,
            float           beta2,
            cudaStream_t    streamId = 0
        );



// -------------------------------------
//  Utility
// -------------------------------------

template<typename T>
BBCU_DLL_EXPORT int bbcu_Tensor_IsnNan
        (
            int             *dev_result,
            T   const       *dev_buf,
            int             size,
            cudaStream_t    streamId=0
        );

template<typename T>
BBCU_DLL_EXPORT int bbcu_FrameBuf_IsnNan
        (
            int             *dev_result,
            T   const       *dev_buf,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId=0
        );


template<typename T>
BBCU_DLL_EXPORT int bbcu_Tensor_Min
        (
            T               *dev_result,
            T   const       *dev_buf,
            int             size,
            cudaStream_t    streamId=0
        );


template<typename T>
BBCU_DLL_EXPORT int bbcu_FrameBuf_Min
        (
            T               *dev_result,
            T   const       *dev_buf,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId=0
        );


template<typename T>
BBCU_DLL_EXPORT int bbcu_Tensor_Max
        (
            T               *dev_result,
            T   const       *dev_buf,
            int             size,
            cudaStream_t    streamId=0
        );


template<typename T>
BBCU_DLL_EXPORT int bbcu_FrameBuf_Max
        (
            T               *dev_result,
            T   const       *dev_buf,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId=0
        );



template<typename T>
BBCU_DLL_EXPORT int bbcu_Tensor_Quantize
        (
            T               *dev_result,
            T   const       *dev_buf,
            T               lo,
            T               hi,
            T               scale,
            int             size,
            cudaStream_t    streamId=0
        );


template<typename T>
BBCU_DLL_EXPORT int bbcu_FrameBuf_Quantize
        (
            T               *dev_dst,
            T   const       *dev_src,
            T               lo,
            T               hi,
            T               scale,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId=0
        );



#if defined(__cplusplus) && defined(BBCU_DLL)
}
#endif


// end of file
