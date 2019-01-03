// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once


#include <iostream>
#include <sstream>
#include <iomanip>

#include "bb/NeuralNetLayer.h"
#include "bb/NeuralNetBinaryLut.h"
#include "bb/NeuralNetLoweringConvolution.h"


namespace bb {

// Jelly用 Convolution 出力
void NeuralNetBinaryConvolutionJelly(std::ostream& os, std::string module_name, std::string mlp_name, int c, int n, int m)
{
	os << "\n\n\n";
	os << "module " << module_name << "\n";

	os << R"(
		#(
			parameter	USER_WIDTH = 0,
			parameter	MAX_X_NUM  = 1024,
			parameter	USE_VALID  = 0,
			parameter	RAM_TYPE   = "block",
			parameter	DEVICE     = "rtl",
			
			parameter	USER_BITS  = USER_WIDTH > 0 ? USER_WIDTH : 1
		)
		(
			input	wire							reset,
			input	wire							clk,
			input	wire							cke,
			
			input	wire							s_img_line_first,
			input	wire							s_img_line_last,
			input	wire							s_img_pixel_first,
			input	wire							s_img_pixel_last,
			input	wire							s_img_de,
			input	wire	[USER_BITS-1:0]			s_img_user,
			input	wire	[0:0]					s_img_data,
			input	wire							s_img_valid,
			
			output	wire							m_img_line_first,
			output	wire							m_img_line_last,
			output	wire							m_img_pixel_first,
			output	wire							m_img_pixel_last,
			output	wire							m_img_de,
			output	wire	[USER_BITS-1:0]			m_img_user,
			output	wire	[0:0]					m_img_data,
			output	wire							m_img_valid
		);
)";

	os << "\tlocalparam	C  = " << c << ";\n";
	os << "\tlocalparam	N  = " << n << ";\n";
	os << "\tlocalparam	M  = " << m << ";\n";
	
	os << R"(
	localparam	NC = N / 2;
	localparam	MC = M / M;
		
	wire							img_blk_line_first;
	wire							img_blk_line_last;
	wire							img_blk_pixel_first;
	wire							img_blk_pixel_last;
	wire							img_blk_de;
	wire	[USER_BITS-1:0]			img_blk_user;
	wire	[N*M*C-1:0]				img_blk_data;
	wire							img_blk_valid;
	
	jelly_img_blk_buffer
			#(
				.USER_WIDTH			(USER_WIDTH),
				.DATA_WIDTH			(1),
				.PIXEL_NUM			(N),
				.LINE_NUM			(N),
				.PIXEL_CENTER		(NC),
				.LINE_CENTER		(MC),
				.MAX_X_NUM			(MAX_X_NUM),
				.RAM_TYPE			(RAM_TYPE),
				.BORDER_MODE		("CONSTANT"),
				.BORDER_VALUE		({(N*M){1'b0}})
			)
		i_img_blk_buffer
			(
				.reset				(reset),
				.clk				(clk),
				.cke				(cke),
				
				.s_img_line_first	(s_img_line_first),
				.s_img_line_last	(s_img_line_last),
				.s_img_pixel_first	(s_img_pixel_first),
				.s_img_pixel_last	(s_img_pixel_last),
				.s_img_de			(s_img_de),
				.s_img_user			(s_img_user),
				.s_img_data			(s_img_data),
				.s_img_valid		(s_img_valid),
				
				.m_img_line_first	(img_blk_line_first),
				.m_img_line_last	(img_blk_line_last),
				.m_img_pixel_first	(img_blk_pixel_first),
				.m_img_pixel_last	(img_blk_pixel_last),
				.m_img_de			(img_blk_de),
				.m_img_user			(img_blk_user),
				.m_img_data			(img_blk_data),
				.m_img_valid		(img_blk_valid)
			);
)";

	os << "\n\n";
	os << "\t" << module_name << "\n";
	os << R"(
		#(
			.USER_WIDTH	(USER_BITS + 6),
			.DEVICE		(DEVICE)
		)
	i_mlp
		(
			.reset		(reset),
			.clk		(clk),
			.cke		(cke),
			
			.in_user	({
							img_blk_user,
							img_blk_line_first,
							img_blk_line_last,
							img_blk_pixel_first,
							img_blk_pixel_last,
							img_blk_de,
						}),
			.in_data	(img_blk_data),
			.in_valid	(img_blk_valid),
			
			.out_user	({
							m_img_user,
							m_img_line_first,
							m_img_line_last,
							m_img_pixel_first,
							m_img_pixel_last,
							m_img_de
						}),
			.out_data	(m_img_data),
			.out_valid	(m_img_valid),
		);"
)";


}



template <typename T = float>
void NeuralNetBinaryConvolutionJelly(std::ostream& os, std::string module_name, NeuralNetLoweringConvolution<T> layer)
{

}


template <typename T = float>
void NeuralNetBinaryCnnAxi4s(std::ostream& os, std::string module_name, std::vector< const NeuralNetLayer<T>* > layers_src)
{
	std::vector< const NeuralNetLayer<T>* >	layers;
	std::vector< std::string >						layer_name;

	int	layer_num = 0;
	for (auto l : layers_src) {
		if (typeid(*l) == typeid(const NeuralNetLoweringConvolution<T>) ) {
			auto layer = (const NeuralNetLoweringConvolution<T> *)l;

			std::stringstream	ss;
			ss << module_name << "_l" << layer_num++;

			NeuralNetLoweringConvolution<T>
		}
	}

}


}