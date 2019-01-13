
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
			int				input_node_size,
			int				output_node_size,
			int				frame_size,
			const float*	in_sig,
			float*			out_sig,
			const int*		input_index,
			const float*	hidden_W,
			const float*	hidden_b,
			const float*	output_W,
			const float*	output_b
		);


}
