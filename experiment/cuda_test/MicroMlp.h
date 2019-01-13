
#pragma once


int MicroMlp6x16_Forward
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