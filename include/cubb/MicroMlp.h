
#pragma once

/*
#ifdef DLL_EXPORT
#define CUBB_DLL_EXPORT __declspec(dllexport) 
#else
#define CUBB_DLL_EXPORT __declspec(dllimport) 
#endif
*/

#define	CUBB_DLL_EXPORT


extern "C" {

CUBB_DLL_EXPORT int MicroMlp6x16_Forward
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


CUBB_DLL_EXPORT int cubb_MicroMlp6x16_Forward
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

}
