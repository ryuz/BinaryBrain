
#pragma once

#include <assert.h>

#include "cuda_runtime.h"


/*
#ifdef DLL_EXPORT
#define CUBB_DLL_EXPORT __declspec(dllexport) 
#else
#define CUBB_DLL_EXPORT __declspec(dllimport) 
#endif
*/

#define	CUBB_DLL_EXPORT	/**/


extern "C" {


#define BBCU_ASSERT(x)          assert(x)
#define BBCU_DEBUG_ASSERT(x)    assert(x)



// -------------------------------------
//  Vector Operation
// -------------------------------------

CUBB_DLL_EXPORT void bbcu_SetHostOnly(bool hostOnly);
CUBB_DLL_EXPORT bool bbcu_IsHostOnly(void);
CUBB_DLL_EXPORT bool bbcu_IsDeviceAvailable(void);



// -------------------------------------
//  Vector Operation
// -------------------------------------

// dst = a * src0 + b * src1 + c
CUBB_DLL_EXPORT int bbcu_fp32_Vector_add_ex
        (
            float*		    dev_dst,
            const float*	dev_src0,
            const float*	dev_src1,
            float			a,
            float			b,
            float			c,
            int				size,
            cudaStream_t	streamId = 0
        );
    
// dst = a * src0 * src1 + b
CUBB_DLL_EXPORT int bbcu_fp32_Vector_mul_ex
        (
            float*			dev_dst,
            const float*	dev_src0,
            const float*	dev_src1,
            float			a,
            float			b,
            int				size,
            cudaStream_t	streamId = 0
        );

// dst = (a * src0 + b) / (c * src1 + d)
CUBB_DLL_EXPORT int bbcu_fp32_Vector_div_ex(
            float           *dev_dst,
            float const     *dev_src0,
            float const     *dev_src1,
            float	        a,
            float	        b,
            float	        c,
            float	        d,
			int				size,
            cudaStream_t	streamId = 0
		);

// dst = sqrt(src)
CUBB_DLL_EXPORT int bbcu_fp32_Vector_sqrt(
            float           *dev_dst,
            float const     *dev_src,
			int				size,
            cudaStream_t	streamId = 0
		);


// dst = exp(src)
CUBB_DLL_EXPORT int bbcu_fp32_Vector_exp(
            float           *dev_dst,
            float const     *dev_src,
			int				size,
            cudaStream_t	streamId = 0
		);


//  min(ベクトル同士)
CUBB_DLL_EXPORT int bbcu_fp32_Vector_min(
            float           *dev_dst,
            float const     *dev_src0,
            float const     *dev_src1,
			int				size,
            cudaStream_t	streamId = 0
		);

// min(係数)
CUBB_DLL_EXPORT int bbcu_fp32_Vector_min_v(
            float           *dev_dst,
            float const     *dev_src,
            float           a,
			int				size,
            cudaStream_t	streamId = 0
		);


//  max(ベクトル同士)
CUBB_DLL_EXPORT int bbcu_fp32_Vector_max(
            float           *dev_dst,
            float const     *dev_src0,
            float const     *dev_src1,
			int				size,
            cudaStream_t	streamId = 0
		);

// max(係数)
CUBB_DLL_EXPORT int bbcu_fp32_Vector_max_v(
            float           *dev_dst,
            float const     *dev_src,
            float           a,
			int				size,
            cudaStream_t	streamId = 0
		);


// clamp
CUBB_DLL_EXPORT int bbcu_fp32_Vector_clamp(
            float           *dev_dst,
            float const     *dev_src,
            float           lo,
            float           hi,
			int				size,
            cudaStream_t	streamId = 0
		);




// Horizontal Sum
CUBB_DLL_EXPORT	int bbcu_fp32_HorizontalSum
        (
            const float*	dev_src,
            float*			dev_dst,
            int				x_size,
            int				y_size,
            cudaStream_t	streamId = 0
        );


// Horizontal MeanVar
CUBB_DLL_EXPORT int bbcu_fp32_HorizontalMeanVar
		(
			const float*	dev_src,
			float*			dev_mean,
			float*			dev_variance,
			int				node_size,
			int				frame_size,
			int				frame_stride,
			cudaStream_t	streamId = 0
		);


// -------------------------------------
//  MicroMlp
// -------------------------------------

CUBB_DLL_EXPORT int bbcu_fp32_MicroMlp6x16_Forward
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
        );

CUBB_DLL_EXPORT int bbcu_fp32_MicroMlp6x16_Backward(
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
        );



// -------------------------------------
//  BatchNormalization
// -------------------------------------

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
			cudaStream_t    streamId = 0
		);



// -------------------------------------
//  Im2Col
// -------------------------------------

CUBB_DLL_EXPORT	int cubb_fp32_Im2Col_Forward
        (
            const float*	input_sig_dev_buf,
            int				input_frame_size,
            int				input_frame_stride,
            int				input_w_size,
            int				input_h_size,
            int				input_c_size,
            float*			output_sig_dev_buf,
            int				output_frame_stride,
            int				filter_w_size,
            int				filter_h_size,
            cudaStream_t    streamId = 0   
        );

CUBB_DLL_EXPORT int cubb_fp32_Im2Col_Backward
        (
			float*			input_grad_dev_buf,
			int				input_frame_size,
			int				input_frame_stride,
			int				input_w_size,
			int				input_h_size,
			int				input_c_size,
			const float*	out_grad_dev_buf,
			int				output_frame_stride,			
            int				filter_w_size,
			int				filter_h_size,            
            cudaStream_t	streamId = 0
        );



/// ReLU
CUBB_DLL_EXPORT int cubb_fp32_ReLU_Forward
		(
			const float*	dev_x_buf,
			float*			dev_y_buf,
			int				frame_size,
			int				frame_stride,
			int				node_size,
            cudaStream_t    streamId = 0
        );

CUBB_DLL_EXPORT int cubb_fp32_ReLU_Backward
		(
			const float*	dev_x_buf,
			const float*	dev_dy_buf,
			float*			dev_dx_buf,
			int				frame_size,
			int				frame_stride,
			int				node_size,
            cudaStream_t    streamId = 0
        );


// Binarize
CUBB_DLL_EXPORT int cubb_fp32_Binarize_Forward
		(
			const float*	dev_x_buf,
			float*			dev_y_buf,
			int				frame_size,
			int				frame_stride,
			int				node_size,
            cudaStream_t    streamId = 0
        );

CUBB_DLL_EXPORT int cubb_fp32_HardTanh_Backward
		(
			const float*	dev_x_buf,
			const float*	dev_dy_buf,
			float*			dev_dx_buf,
			int				frame_size,
			int				frame_stride,
			int				node_size,
            cudaStream_t    streamId = 0
        );



}


// end of file
