
#pragma once

#include <assert.h>

#include "cuda_runtime.h"
#include "bbcu/bbcu_util.h"

/*
#ifdef DLL_EXPORT
#define BBCU_DLL_EXPORT __declspec(dllexport) 
#else
#define BBCU_DLL_EXPORT __declspec(dllimport) 
#endif
*/

#define BBCU_DLL_EXPORT /**/


extern "C" {


#define BBCU_ASSERT(x)          assert(x)
#define BBCU_DEBUG_ASSERT(x)    assert(x)



// -------------------------------------
//  Vector Operation
// -------------------------------------

BBCU_DLL_EXPORT void bbcu_SetHostOnly(bool hostOnly);
BBCU_DLL_EXPORT bool bbcu_IsHostOnly(void);
BBCU_DLL_EXPORT bool bbcu_IsDeviceAvailable(void);



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


int bbcu_fp32_MatrixRowwiseSetVector
        (
            const float*    dev_x_vec,
            float*          dev_y_mat,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId = 0
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



// -------------------------------------
//  StochasticLut
// -------------------------------------


int bbcu_fp32_StochasticLut6_Forward
        (
            const float     *dev_x_buf,
            float           *dev_y_buf,
            int   const     *dev_input_index,
            float const     *dev_W,
            int             node_size,
            int             frame_size,
            int             frame_stride,
            int             binary_mode,
            cudaStream_t    streamId = 0
        );

int bbcu_fp32_StochasticLut6_Backward(
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
            int             binary_mode,
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
            float const     *dev_y_buf,
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

BBCU_DLL_EXPORT int bbcu_fp32_StochasticBatchNormalization_ForwardInference
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

BBCU_DLL_EXPORT int bbcu_fp32_StochasticBatchNormalization_Backward
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


// -------------------------------------
//  Im2Col
// -------------------------------------

BBCU_DLL_EXPORT int bbcu_fp32_Im2Col_Forward
        (
            float const     *dev_x_buf,
            float           *dev_y_buf,
            int             input_frame_size,
            int             input_frame_stride,
            int             input_w_size,
            int             input_h_size,
            int             input_c_size,
            int             output_frame_stride,
            int             filter_w_size,
            int             filter_h_size,
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


BBCU_DLL_EXPORT int bbcu_fp32_Im2Col_Backward
        (
            float const     *dev_fy_buf,
            float           *dev_dx_buf,
            int             input_frame_size,
            int             input_frame_stride,
            int             input_w_size,
            int             input_h_size,
            int             input_c_size,
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

BBCU_DLL_EXPORT int bbcu_bit_Im2Col_Forward
        (
            int const       *dev_x_buf,
            int             *dev_y_buf,
            int             input_frame_size,
            int             input_frame_stride,
            int             input_w_size,
            int             input_h_size,
            int             input_c_size,
            int             output_frame_stride,
            int             filter_w_size,
            int             filter_h_size,
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
            int             node_size,
            int             frame_size,
            int             frame_stride,
            cudaStream_t    streamId = 0
        );



// -------------------------------------
//  Hard-Tanh
// -------------------------------------

BBCU_DLL_EXPORT int bbcu_fp32_HardTanh_Forward
        (
            float const *   dev_x_buf,
            float*          dev_y_buf,
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
//  BinaryToReal
// -------------------------------------

BBCU_DLL_EXPORT int bbcu_fp32_BinaryToReal_Forward
        (
            const float*    dev_x_buf,
            float*          dev_y_buf,
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
            const float*    dev_dy_buf,
            float*          dev_dx_buf,
            int             node_mux_size,
            int             frame_mux_size,
            int             y_node_size,
            int             x_frame_stride,
            int             y_frame_size,
            int             y_frame_stride,
            cudaStream_t    streamId = 0
        );

}



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

// end of file
