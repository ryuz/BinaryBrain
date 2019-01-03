// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once


#include <iostream>
#include <iomanip>
#include <vector>
#include <sstream>


#include "bb/NeuralNet.h"
#include "bb/NeuralNetGroup.h"
#include "bb/NeuralNetBinaryLut.h"
#include "bb/NeuralNetLoweringConvolution.h"


namespace bb {


// NeuralNetBinaryLut Verilog 出力
template <typename T = float>
void OutputVerilogBinaryLut(std::ostream& os, std::string module_name, NeuralNetBinaryLut<T>& lut)
{
	int		lut_input_size = lut.GetLutInputSize();
	int		lut_table_size = lut.GetLutTableSize();
	INDEX	node_size      = lut.GetOutputNodeSize();
	
	
	// モジュール出力
	os <<
		"\n"
		"\n"
		"module " << module_name << "\n"
		"        #(\n"
		"            parameter DEVICE = \"RTL\"\n"
		"        )\n"
		"        (\n"
		"            input  wire         reset,\n"
		"            input  wire         clk,\n"
		"            input  wire         cke,\n"
		"            \n"
		"            input  wire [" << (lut.GetInputNodeSize() - 1) << ":0]  in_data,\n"
		"            output wire [" << (lut.GetOutputNodeSize() - 1) << ":0]  out_data\n"
		"        );\n"
		"\n";


	for (INDEX node = 0; node < node_size; node++) {
		// LUT 出力
		os <<
			"\n"
			"// LUT : " << node << "\n"
			"\n"
			"wire lut_" << node << "_out;\n"
			"\n"
			"bb_lut\n"
			"        #(\n"
			"            .N(" << lut_input_size << "),\n"
			"            .INIT(" << lut_table_size << "'b";

		for (int bit = lut_table_size - 1; bit >= 0; --bit ) {
			os << (lut.GetLutTable(node, bit) ? "1" : "0");
		}
		os <<
			"),\n"
			"            .DEVICE(DEVICE)\n";

		os <<
			"        )\n"
			"    i_lut_" << node << "\n"
			"        (\n"
			"            .in_data({\n";

		for (int bit = lut_input_size - 1; bit >= 1; --bit) {
			os <<
				"                         in_data[" << lut.GetLutInput(node, bit) << "],\n";
		}
		os <<
			"                         in_data[" << lut.GetLutInput(node, 0) << "]\n"
			"                    }),\n"
			"            .out_data(lut_" << node << "_out)\n"
			"        );\n"
			"\n";

		os <<
			"reg   lut_" << node << "_ff;\n"
			"always @(posedge clk) begin\n"
			"    if ( reset ) begin\n"
			"        lut_" << node << "_ff <= 1'b0;\n"
			"    end\n"
			"    else if ( cke ) begin\n"
			"        lut_" << node << "_ff <= lut_" << node << "_out;\n"
			"    end\n"
			"end\n"
			"\n"
			"assign out_data[" << node << "] = lut_" << node << "_ff;\n"
		"\n";

		os <<
			"\n"
			"\n";
	}

	os <<
		"endmodule\n";
	os << std::endl;
}



// Verilog 出力
template <typename T = float>
void OutputVerilogLutGroup(std::ostream& os, std::string module_name, NeuralNetGroup<T>& group)
{
	auto layer_size = group.GetSize();
	for (int i = 0; i < layer_size; ++i) {
		if (!dynamic_cast<NeuralNetBinaryLut<T>*>(group[i])) {
			std::cout << "error : dynamic_cast<NeuralNetBinaryLut<T>*>" << std::endl;
			BB_ASSERT(0);
			return;
		}
	}


	std::vector<std::string> sub_modle_name;
	auto first_layer = dynamic_cast<NeuralNetBinaryLut<T>*>(group[0]);
	auto last_layer  = dynamic_cast<NeuralNetBinaryLut<T>*>(group[layer_size - 1]);

	// サブモジュール名生成
	for (int i = 0; i < layer_size; ++i) {
		std::stringstream ss_sub_name;
		ss_sub_name << module_name << "_sub" << i;
		sub_modle_name.push_back(ss_sub_name.str());
	}
	
	// モジュール出力
	os <<
		"\n"
		"\n"
		"module " << module_name << "\n"
		"        #(\n"
		"            parameter USER_WIDTH = 0,\n"
		"            parameter DEVICE     = \"RTL\",\n"
		"            \n"
		"            parameter USER_BITS  = USER_WIDTH > 0 ? USER_WIDTH : 1\n"
		"        )\n"
		"        (\n"
		"            input  wire                  reset,\n"
		"            input  wire                  clk,\n"
		"            input  wire                  cke,\n"
		"            \n"
		"            input  wire [USER_BITS-1:0]  in_user,\n"
		"            input  wire [" << std::setw(9) << first_layer->GetInputNodeSize() << "-1:0]  in_data,\n"
		"            input  wire                  in_valid,\n"
		"            \n"
		"            output wire [USER_BITS-1:0]  out_user,\n"
		"            output wire [" << std::setw(9) << last_layer->GetOutputNodeSize() << "-1:0]  out_data,\n"
		"            output wire                  out_valid\n"
		"        );\n"
		"\n\n";

	for (int i = 0; i < layer_size; ++i) {
		auto layer = dynamic_cast<NeuralNetBinaryLut<T>*>(group[i]);

		os
			<< "reg   [USER_BITS-1:0]  layer" << i << "_user;\n"
			<< "wire  [" << std::setw(9) << layer->GetOutputNodeSize() << "-1:0]  layer" << i << "_data;\n"
			<< "reg                    layer" << i << "_valid;\n"
			<< "\n"
			<< sub_modle_name[i] << "\n"
			<< "        #(\n"
			<< "            .DEVICE     (DEVICE)\n"
			<< "        )\n"
			<< "    i_" << sub_modle_name[i] << "\n"
			<< "        (\n"
			<< "            .reset      (reset),\n"
			<< "            .clk        (clk),\n"
			<< "            .cke        (cke),\n"
			<< "            \n";
		if (i == 0) {
			os << "            .in_data    (in_data),\n";
		}
		else {
			os << "            .in_data    (layer" << (i - 1) << "_data),\n";
		}
		os
			<< "            .out_data   (layer" << i << "_data)\n"
			<< "         );\n"
			<< "\n"
			<< "always @(posedge clk) begin\n"
			<< "    if ( reset ) begin\n"
			<< "        layer" << i << "_user  <= {USER_BITS{1'bx}};\n"
			<< "        layer" << i << "_valid <= 1'b0;\n"
			<< "    end\n"
			<< "    else if ( cke ) begin\n";
		if (i == 0) {
			os
				<< "        layer" << i << "_user  <= in_user;\n"
				<< "        layer" << i << "_valid <= in_valid;\n";
		}
		else {
			os
				<< "        layer" << i << "_user  <= layer" << (i - 1) << "_user;\n"
				<< "        layer" << i << "_valid <= layer" << (i - 1) << "_valid;\n";
		}
		os
			<< "    end\n"
			<< "end\n"
			<< "\n\n";
	}

	os
		<< "assign out_data  = layer" << (layer_size - 1) << "_data;\n"
		<< "assign out_user  = layer" << (layer_size - 1) << "_user;\n"
		<< "assign out_valid = layer" << (layer_size - 1) << "_valid;\n"
		<< "\n"
		<< "endmodule\n"
		<< "\n\n";
	

	// サブモジュール出力
	for (int i = 0; i < layer_size; ++i) {
		auto lut = dynamic_cast<NeuralNetBinaryLut<T>*>(group[i]);

		OutputVerilogBinaryLut<T>(os, sub_modle_name[i], *lut);
	}
}



// Convolution 出力
void OutputVerilogConvolution(std::ostream& os, std::string module_name, std::string mlp_name, int c, int n, int m)
{
	os << "\n\n\n";
	os << "module " << module_name << "\n";

	os << R"(
		#(
			parameter	USER_WIDTH = 0,
			parameter	MAX_X_NUM  = 1024,
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
	os << "\t" << mlp_name << "\n";
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
								img_blk_de
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
				.out_valid	(m_img_valid)
			);


endmodule
)";


}




template <typename T = float, typename ST = bool>
void OutputVerilogLoweringConvolution(std::ostream& os, std::string module_name, NeuralNetLoweringConvolution<ST, T>& conv)
{
	// group取得
	auto grop = dynamic_cast<NeuralNetGroup<T> *>(conv.GetLayer());
	if ( !grop ) {
		std::cout << "error : dynamic_cast<NeuralNetGroup<T> *>" << std::endl;
		BB_ASSERT(0);
		return;
	}

	std::string mlp_name = module_name + "_mlp";

	int c = conv.GetOutputChannel();
	int n = conv.GetFilterHeight();
	int m = conv.GetFilterWidth();

	OutputVerilogConvolution(os, module_name, mlp_name, c, n, m);
	OutputVerilogLutGroup(os, mlp_name, *grop);
}




template <typename T = float, typename ST = bool>
void OutputVerilogCnnAxi4s(std::ostream& os, std::string module_name, std::vector< NeuralNetFilter2d<T>* > layers)
{
	int	 layer_size = (int)layers.size();
	auto fisrt_layer = layers[0];
	auto last_layer = layers[layer_size - 1];

	os << "module " << module_name << "\n"; 
	os << R"(
		#(
			parameter	TUSER_WIDTH    = 1,
			parameter	IMG_X_WIDTH    = 10,
			parameter	IMG_Y_WIDTH    = 9,
			parameter	IMG_Y_NUM      = 480,
			parameter	BLANK_Y_WIDTH  = 8,
			parameter	INIT_Y_NUM     = IMG_Y_NUM,
			parameter	FIFO_PTR_WIDTH = 9,
			parameter	FIFO_RAM_TYPE  = "block",
			parameter	IMG_CKE_BUFG   = 0,
			parameter	DEVICE         = "rtl",
)";

	os << "			parameter	S_TDATA_WIDTH  = " << fisrt_layer->GetInputNodeSize() << ",\n";
	os << "			parameter	M_TDATA_WIDTH  = " << last_layer->GetOutputNodeSize();

	os << R"(
		)
		(
			input	wire								reset,
			input	wire								clk,
			
			input	wire	[BLANK_Y_WIDTH-1:0]			param_blank_num,
			
			input	wire	[TUSER_WIDTH-1:0]			s_axi4s_tuser,
			input	wire								s_axi4s_tlast,
			input	wire	[S_TDATA_WIDTH-1:0]			s_axi4s_tdata,
			input	wire								s_axi4s_tvalid,
			output	wire								s_axi4s_tready,
			
			output	wire	[TUSER_WIDTH-1:0]			m_axi4s_tuser,
			output	wire								m_axi4s_tlast,
			output	wire	[M_TDATA_WIDTH-1:0]			m_axi4s_tdata,
			output	wire								m_axi4s_tvalid,
			input	wire								m_axi4s_tready
		);
)";

	os << R"(

	localparam	USER_WIDTH = TUSER_WIDTH > 1 ? TUSER_WIDTH - 1 : 1;

	wire								cke;
	
	wire								src_img_line_first;
	wire								src_img_line_last;
	wire								src_img_pixel_first;
	wire								src_img_pixel_last;
	wire								src_img_de;
	wire	[USER_WIDTH-1:0]			src_img_user;
	wire	[S_TDATA_WIDTH-1:0]			src_img_data;
	wire								src_img_valid;
	
	wire								sink_img_line_first;
	wire								sink_img_line_last;
	wire								sink_img_pixel_first;
	wire								sink_img_pixel_last;
	wire								sink_img_de;
	wire	[USER_WIDTH-1:0]			sink_img_user;
	wire	[M_TDATA_WIDTH-1:0]			sink_img_data;
	wire								sink_img_valid;
	
	jelly_axi4s_img
			#(
				.TUSER_WIDTH			(TUSER_WIDTH),
				.S_TDATA_WIDTH			(S_TDATA_WIDTH),
				.M_TDATA_WIDTH			(M_TDATA_WIDTH),
				.IMG_X_WIDTH			(IMG_X_WIDTH),
				.IMG_Y_WIDTH			(IMG_Y_WIDTH),
				.IMG_Y_NUM				(IMG_Y_NUM),
				.USE_DE					(1),
				.USE_VALID				(1),
				.BLANK_Y_WIDTH			(BLANK_Y_WIDTH),
				.INIT_Y_NUM				(INIT_Y_NUM),
				.FIFO_PTR_WIDTH			(FIFO_PTR_WIDTH),
				.FIFO_RAM_TYPE			(FIFO_RAM_TYPE),
				.IMG_CKE_BUFG			(IMG_CKE_BUFG)
			)
		i_axi4s_img
			(
				.reset					(reset),
				.clk					(clk),
				
				.param_blank_num		(param_blank_num),
				
				.s_axi4s_tuser			(s_axi4s_tuser),
				.s_axi4s_tlast			(s_axi4s_tlast),
				.s_axi4s_tdata			(s_axi4s_tdata),
				.s_axi4s_tvalid			(s_axi4s_tvalid),
				.s_axi4s_tready			(s_axi4s_tready),
				
				.m_axi4s_tuser			(m_axi4s_tuser),
				.m_axi4s_tlast			(m_axi4s_tlast),
				.m_axi4s_tdata			(m_axi4s_tdata),
				.m_axi4s_tvalid			(m_axi4s_tvalid),
				.m_axi4s_tready			(m_axi4s_tready),
				
				
				.img_cke				(cke),
				
				.src_img_line_first		(src_img_line_first),
				.src_img_line_last		(src_img_line_last),
				.src_img_pixel_first	(src_img_pixel_first),
				.src_img_pixel_last		(src_img_pixel_last),
				.src_img_de				(src_img_de),
				.src_img_user			(src_img_user),
				.src_img_data			(src_img_data),
				.src_img_valid			(src_img_valid),
				
				.sink_img_line_first	(sink_img_line_first),
				.sink_img_line_last		(sink_img_line_last),
				.sink_img_pixel_first	(sink_img_pixel_first),
				.sink_img_pixel_last	(sink_img_pixel_last),
				.sink_img_de			(sink_img_de),
				.sink_img_user			(sink_img_user),
				.sink_img_data			(sink_img_data),
				.sink_img_valid			(sink_img_valid)
			);
	
	
)";

	os << "\tlocalparam DATA0_WIDTH = " << fisrt_layer->GetInputChannel() << " * " << fisrt_layer->GetFilterHeight() << " * " << fisrt_layer->GetFilterWidth() << ";\n";
	for ( int i = 0; i < layer_size; ++i ) {
		os << "\tlocalparam DATA" << i+1 << "_WIDTH = " << layers[i]->GetInputChannel() << " * " << layers[i]->GetFilterHeight() << " * " << layers[i]->GetFilterWidth() << ";\n";
	}
	os << "\t\n";
	
	for ( int i = 0; i < layer_size; ++i ) {
		os << "\t\n";
		os << "\twire							img" << i << "_line_first;\n";
		os << "\twire							img" << i << "_line_last;\n";
		os << "\twire							img" << i << "_pixel_first;\n";
		os << "\twire							img" << i << "_pixel_last;\n";
		os << "\twire							img" << i << "_de;\n";
		os << "\twire	[USER_WIDTH-1:0]		img" << i << "_user;\n";
		os << "\twire	[DATA" << i << "_WIDTH-1:0]		img" << i << "_data;\n";
		os << "\twire							img" << i << "_valid;\n";
	}


	for ( int i = 0; i < layer_size; ++i ) {
		os << "\n\n";

		auto layer = layers[i];
		auto cnv = dynamic_cast<NeuralNetLoweringConvolution<ST, T>*>(layer);
		auto pol = dynamic_cast<NeuralNetMaxPooling<ST, T, T>*>(layer);
		if ( cnv ) {
			os << "\t" << module_name << "_layer" << i << "\n";
			os << "\t\t\t#(\n";
			os << "\t\t\t\t.USER_WIDTH				(USER_WIDTH),\n";
			os << "\t\t\t\t.MAX_X_NUM				(MAX_X_NUM),\n";
			os << "\t\t\t\t.RAM_TYPE				(RAM_TYPE),\n";
			os << "\t\t\t\t.DEVICE					(DEVICE)\n";
			os << "\t\t\t)\n";
			os << "\t\ti_" << module_name << "_l" << i << "\n";
		}
		else if (pol) {
			os << "\t" << "jelly_img_dnn_maxpol" << "\n";
			os << "\t\t\t#(\n";
			os << "\t\t\t\t.C						(" << pol->GetOutputChannel() << "),\n";
			os << "\t\t\t\t.N						(" << pol->GetFilterWidth() << "),\n";
			os << "\t\t\t\t.M						(" << pol->GetFilterHeight() << "),\n";
			os << "\t\t\t\t.USER_WIDTH				(USER_WIDTH),\n";
			os << "\t\t\t\t.MAX_X_NUM				(MAX_X_NUM),\n";
			os << "\t\t\t\t.RAM_TYPE				(RAM_TYPE)\n";
			os << "\t\t\t)\n";
			os << "\t\ti_" << "i_img_dnn_maxpol" << "_l" << i << "\n";
		}
		else {
			std::cout << "error" << std::endl;
			BB_ASSERT(0);
			return;
		}

		os << "\t\t\t(\n";
		os << "\t\t\t\t.reset					(reset),\n";
		os << "\t\t\t\t.clk					(clk),\n";
		os << "\t\t\t\t.cke					(cke),\n";
		os << "\t\t\t\t\n";
		os << "\t\t\t\t.s_img_line_first		(img" << i << "_line_first),\n";
		os << "\t\t\t\t.s_img_line_last		(img" << i << "_line_last),\n";
		os << "\t\t\t\t.s_img_pixel_first		(img" << i << "_pixel_first),\n";
		os << "\t\t\t\t.s_img_pixel_last		(img" << i << "_pixel_last),\n";
		os << "\t\t\t\t.s_img_de				(img" << i << "_de),\n";
		os << "\t\t\t\t.s_img_user				(img" << i << "_user),\n";
		os << "\t\t\t\t.s_img_data				(img" << i << "_data),\n";
		os << "\t\t\t\t.s_img_valid			(img" << i << "_valid),\n";
		os << "\t\t\t\t\n";
		os << "\t\t\t\t.m_img_line_first		(img" << i+1 << "_line_first),\n";
		os << "\t\t\t\t.m_img_line_last		(img" << i+1 << "_line_last),\n";
		os << "\t\t\t\t.m_img_pixel_first		(img" << i+1 << "_pixel_first),\n";
		os << "\t\t\t\t.m_img_pixel_last		(img" << i+1 << "_pixel_last),\n";
		os << "\t\t\t\t.m_img_de				(img" << i+1 << "_de),\n";
		os << "\t\t\t\t.m_img_user				(img" << i+1 << "_user),\n";
		os << "\t\t\t\t.m_img_data				(img" << i+1 << "_data),\n";
		os << "\t\t\t\t.m_img_valid			(img" << i+1 << "_valid)\n";
		os << "\t\t\t);\n";
	}

	os << "\t\n";
	os << "\t\n";
	os << "\tassign img" << 0 << "_line_first  = src_img_line_first;\n";
	os << "\tassign img" << 0 << "_line_last   = src_img_line_last;\n";
	os << "\tassign img" << 0 << "_pixel_first = src_img_pixel_first;\n";
	os << "\tassign img" << 0 << "_pixel_last  = src_img_pixel_last;\n";
	os << "\tassign img" << 0 << "_de          = src_img_de;\n";
	os << "\tassign img" << 0 << "_user        = src_img_user;\n";
	os << "\tassign img" << 0 << "_data        = src_img_data;\n";
	os << "\tassign img" << 0 << "_valid       = src_img_valid;\n";
	os << "\t\n";
	os << "\tassign sink_img_line_first  = img" << layer_size << "_line_first;\n";
	os << "\tassign sink_img_line_last   = img" << layer_size << "_line_last;\n";
	os << "\tassign sink_img_pixel_first = img" << layer_size << "_pixel_first;\n";
	os << "\tassign sink_img_pixel_last  = img" << layer_size << "_pixel_last;\n";
	os << "\tassign sink_img_de          = img" << layer_size << "_de;\n";
	os << "\tassign sink_img_user        = img" << layer_size << "_user;\n";
	os << "\tassign sink_img_data        = img" << layer_size << "_data;\n";
	os << "\tassign sink_img_valid       = img" << layer_size << "_valid;\n";

	os << "\t\n";
	os << "\t\n";
	os << "endmodule\n\n";

	for ( int i = 0; i < layer_size; ++i ) {
		auto layer = layers[i];
		auto cnv = dynamic_cast<NeuralNetLoweringConvolution<ST, T>*>(layer);
		if ( cnv ) {
			std::stringstream ss;
			ss << module_name << "_l" << i;
			OutputVerilogLoweringConvolution(os, ss.str(), *cnv);
		}
	}
}


}