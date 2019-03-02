
#pragma once

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


// -------------------------------------
//  Vector Operation
// -------------------------------------

// dst = a * src0 + b * src1 + c
CUBB_DLL_EXPORT int bbcu_Vector_add_ex
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
CUBB_DLL_EXPORT int bbcu_Vector_mul_ex
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
CUBB_DLL_EXPORT int bbcu_Vector_div_ex(
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
CUBB_DLL_EXPORT int bbcu_Vector_sqrt(
            float           *dev_dst,
            float const     *dev_src,
			int				size,
            cudaStream_t	streamId = 0
		);


// dst = exp(src)
CUBB_DLL_EXPORT int bbcu_Vector_exp(
            float           *dev_dst,
            float const     *dev_src,
			int				size,
            cudaStream_t	streamId = 0
		);


// Horizontal Sum
CUBB_DLL_EXPORT	int bbcu_HorizontalSum
        (
            const float*	dev_src,
            float*			dev_dst,
            int				x_size,
            int				y_size,
            cudaStream_t	streamId = 0
        );



// -------------------------------------
//  MicroMlp
// -------------------------------------

CUBB_DLL_EXPORT int bbcu_MicroMlp6x16_Forward
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
            cudaStream_t	streamId = 0
        );

// -------------------------------------
//  Im2Col
// -------------------------------------


CUBB_DLL_EXPORT	int cubb_Im2Col_Forward
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

CUBB_DLL_EXPORT int cubb_Im2Col_Backward
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

/// ---- test code ----



CUBB_DLL_EXPORT int bbcu_eva_HorizontalSum
        (
            const float*	src,
            float*			dst,
            int				x_size,
            int				y_size
        );

CUBB_DLL_EXPORT int bbcu_eva_MicroMlp6x16_Forward
        (
            const float*	in_sig,
            float*			out_sig,
            int				input_node_size,
            int				output_node_size,
            int				frame_size,
            const int*		input_index,
            const float*	hidden_W,
            const float*	hidden_b,
            const float*	output_W,
            const float*	output_b
        );

CUBB_DLL_EXPORT int bbcu_eva_MicroMlp6x16_Backward
        (
            const float*	in_sig_buf,
            float*			in_err_buf,
            float*			out_err_buf,
            int				input_node_size,
            int				output_node_size,
            int				frame_size,
            const int*		input_index,
            const float*	hidden_W,
            const float*	hidden_b,
            float*			hidden_dW,
            float*			hidden_db,
            const float*	output_W,
            const float*	output_b,
            float*			output_dW,
            float*			output_db
        );



CUBB_DLL_EXPORT int bbcu_eva_Im2Col_Forward
        (
            const float*	input_sig,
            int				input_frame_size,
            int     		input_frame_stride,
            int				input_w_size,
            int				input_h_size,
            int				input_c_size,
            float*			output_sig,
            int     		output_frame_stride,
            int				filter_w_size,
            int				filter_h_size
        );

CUBB_DLL_EXPORT int bbcu_eva_Im2Col_Backward
        (
            float*			in_err_buf,
            const float*	out_err_buf,
            int				input_frame_size,
            int				input_w_size,
            int				input_h_size,
            int				input_c_size,
            int				filter_w_size,
            int				filter_h_size
        );


}
