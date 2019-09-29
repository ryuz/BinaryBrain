
#pragma once

#include <assert.h>

#include "cuda_runtime.h"
#include "bb/Assert.h"


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

BBCU_DLL_EXPORT void bbcu_SetHostOnly(bool hostOnly);
BBCU_DLL_EXPORT bool bbcu_IsHostOnly(void);
BBCU_DLL_EXPORT bool bbcu_IsDeviceAvailable(void);



// -------------------------------------
//  Local Heap
// -------------------------------------

BBCU_DLL_EXPORT void  *bbcu_LocalHeap_Malloc(size_t size);
BBCU_DLL_EXPORT void   bbcu_LocalHeap_Free(void* ptr);
BBCU_DLL_EXPORT size_t bbcu_LocalHeap_GetMaxAllocSize(void);


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
//  Binary LUT
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
//  SparseLut
// -------------------------------------

template <int N=6>
BBCU_DLL_EXPORT int bbcu_fp32_SparseLutN_ForwardTraining
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
BBCU_DLL_EXPORT int bbcu_bit_fp32_SparseLutN_ForwardTraining
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
BBCU_DLL_EXPORT int bbcu_fp32_SparseLutN_ForwardInference
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
BBCU_DLL_EXPORT int bbcu_bit_fp32_SparseLutN_ForwardInference
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
BBCU_DLL_EXPORT int bbcu_fp32_SparseLutN_Backward
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
BBCU_DLL_EXPORT int bbcu_bit_fp32_SparseLutN_Backward
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

#if 0
BBCU_DLL_EXPORT int bbcu_fp32_SparseLut6_ForwardTraining
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

BBCU_DLL_EXPORT int bbcu_fp32_SparseLut4_ForwardTraining
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

BBCU_DLL_EXPORT int bbcu_bit_fp32_SparseLut6_ForwardTraining
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

BBCU_DLL_EXPORT int bbcu_bit_fp32_SparseLut4_ForwardTraining
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
BBCU_DLL_EXPORT int bbcu_fp32_SparseLut6_ForwardInference
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

BBCU_DLL_EXPORT int bbcu_fp32_SparseLut4_ForwardInference
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

BBCU_DLL_EXPORT int bbcu_bit_fp32_SparseLut6_ForwardInference
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

BBCU_DLL_EXPORT int bbcu_bit_fp32_SparseLut4_ForwardInference
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

BBCU_DLL_EXPORT int bbcu_fp32_SparseLut6_Backward
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

BBCU_DLL_EXPORT int bbcu_fp32_SparseLut4_Backward
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

BBCU_DLL_EXPORT int bbcu_bit_fp32_SparseLut6_Backward
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

BBCU_DLL_EXPORT int bbcu_bit_fp32_SparseLut4_Backward
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
#endif


// -------------------------------------
//  SparseBinaryLut
// -------------------------------------

BBCU_DLL_EXPORT int bbcu_bit_fp32_SparseBinaryLut6_ForwardTraining
        (
            int   const     *dev_x_buf,
            int             *dev_y_buf,
            int   const     *dev_input_index,
            float const     *dev_W,
            float           *mean_buf,
            float           *rstd_buf,
            float           *running_mean_buf,
            float           *running_var_buf,
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

BBCU_DLL_EXPORT int bbcu_bit_fp32_SparseBinaryLut6_ForwardInference
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

BBCU_DLL_EXPORT int bbcu_bit_fp32_SparseBinaryLut6_Backward
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
            cudaStream_t    streamId=0
    );



// -------------------------------------
//  StochasticLut
// -------------------------------------

BBCU_DLL_EXPORT int bbcu_fp32_StochasticLut6_Forward
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

BBCU_DLL_EXPORT int bbcu_bit_fp32_StochasticLut6_Forward
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

BBCU_DLL_EXPORT int bbcu_bit_bit_fp32_StochasticLut6_Forward
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


BBCU_DLL_EXPORT int bbcu_fp32_StochasticLut6_Backward
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

BBCU_DLL_EXPORT int bbcu_bit_fp32_StochasticLut6_Backward
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

/*
BBCU_DLL_EXPORT int bbcu_fp32_StochasticLut6_Backward
        (
            float const     *dev_x_buf,
            float const     *dev_dy_buf,
            float           *dev_dx_buf,
            float           *dev_dx_tmp,
            int   const     *dev_input_index,
            float const     *dev_W,
            float           *dev_dW,
            int             input_node_size,
            int             output_node_size,
            int             frame_size,
            int             frame_stride,
            int             input_binary,
            int             lut_binarize,
            float           unbinarize_bias,
            cudaStream_t    streamId = 0
        );
*/

/*
BBCU_DLL_EXPORT int bbcu_bit_fp32_StochasticLut6_Backward
        (
            int   const     *dev_x_buf,
            float const     *dev_dy_buf,
            float           *dev_dx_buf,
            float           *dev_dx_tmp,
            int   const     *dev_input_index,
            float const     *dev_W,
            float           *dev_dW,
            int             input_node_size,
            int             output_node_size,
            int             frame_size,
            int             frame_stride,
            int             bit_frame_stride,
            int             lut_binarize,
            float           unbinarize_bias,
            cudaStream_t    streamId = 0
        );
*/

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
            cudaStream_t    streamId = 0
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
            float const     *dev_x_buf,
            float const     *dev_y_buf,
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
            int   const     *dev_y_buf,
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
//  Binarize
// -------------------------------------

BBCU_DLL_EXPORT int bbcu_fp32_Binarize_Forward
        (
            const float*    dev_x_buf,
            float*          dev_y_buf,
            float           binary_th,
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

BBCU_DLL_EXPORT int bbcu_fp32_RealToBinary_Forward
        (
            float const     *dev_x_buf,
            float           *dev_y_buf,
            float           th_offset,
            float           th_step,
            int             modulation_size,
            int             node_size,
            int             x_frame_size,
            int             x_frame_stride,
            int             y_frame_stride,
            cudaStream_t    streamId = 0
        );


BBCU_DLL_EXPORT int bbcu_fp32_bit_no_modulation_RealToBinary_Forward
        (
            float const     *dev_x_buf,
            int             *dev_y_buf,
            float           th,
            int             node_size,
            int             frame_size,
            int             x_frame_stride,
            int             y_frame_stride,
            cudaStream_t    streamId = 0
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
//  LossSoftmaxCrossEntropy
// -------------------------------------

BBCU_DLL_EXPORT int bbcu_fp32_LossSoftmaxCrossEntropy
        (
            float const     *dev_y_buf,
            float const     *dev_t_buf,
            float           *dev_dy_buf,
            float           *dev_loss_buf,
            float           *dev_loss,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            int             batch_size,
            cudaStream_t    streamId = 0
        );


// -------------------------------------
//  AccuracyCategoricalClassification
// -------------------------------------

BBCU_DLL_EXPORT int bbcu_fp32_AccuracyCategoricalClassification
        (
            float const     *dev_y_buf,
            float const     *dev_t_buf,
            int             *dev_accuracy,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId = 0
        );

// -------------------------------------
//  Adam
// -------------------------------------

BBCU_DLL_EXPORT int bbcu_fp32_Adam
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


#if defined(__cplusplus) && defined(BBCU_DLL)
}
#endif


// end of file
