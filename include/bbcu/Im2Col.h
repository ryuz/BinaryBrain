
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

CUBB_DLL_EXPORT	int cubb_Im2Col_Forward
		(
			const float*	dev_in_sig,
			float*			dev_out_sig,
			int				input_frame_size,
			int				input_w_size,
			int				input_h_size,
			int				input_c_size,
			int				filter_w_size,
			int				filter_h_size
		);

CUBB_DLL_EXPORT int Im2Col_Forward
		(
			const float*	in_sig,
			float*			out_sig,
			int				input_frame_size,
			int				input_w_size,
			int				input_h_size,
			int				input_c_size,
			int				filter_w_size,
			int				filter_h_size
		);


}
