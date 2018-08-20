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
#include "NeuralNetBinaryLut.h"


namespace bb {

// Xilinx用 6入力LUT Verilog 出力
template <typename T = float, typename INDEX = size_t>
void NeuralNetBinaryLut6VerilogXilinx(std::string module_name, std::ostream& os, NeuralNetBinaryLut<T, INDEX>& lut)
{
	int		lut_input_size = lut.GetLutInputSize();
	int		lut_table_size = lut.GetLutTableSize();
	INDEX	node_size      = lut.GetOutputNodeSize();

	if (lut_input_size > 6 || lut_table_size > 64) {
		return;		// error
	}

	// モジュール出力
	os <<
		"\n"
		"\n"
		"module " << module_name << "(\n"
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
			"LUT6\n"
			"        #(\n"
			"            .INIT(64'b";

		for (int bit = 0; bit < 64; ++bit ) {
			if (bit < lut_table_size) {
				os << lut.GetLutTable(node, bit) ? "1" : "0";
			}
			else {
				os << "0";
			}
		}

		os << ")\n";

		os <<
			"        )\n"
			"    i_lut6_" << node << "\n"
			"        (\n"
			"            .O  (lut_" << node << "_out),\n"
			"            .I0 (in_data[" << lut.GetLutInput(node, std::min(0, lut_input_size - 1)) << "]),\n"
			"            .I1 (in_data[" << lut.GetLutInput(node, std::min(1, lut_input_size - 1)) << "]),\n"
			"            .I2 (in_data[" << lut.GetLutInput(node, std::min(2, lut_input_size - 1)) << "]),\n"
			"            .I3 (in_data[" << lut.GetLutInput(node, std::min(3, lut_input_size - 1)) << "]),\n"
			"            .I4 (in_data[" << lut.GetLutInput(node, std::min(4, lut_input_size - 1)) << "]),\n"
			"            .I5 (in_data[" << lut.GetLutInput(node, std::min(5, lut_input_size - 1)) << "])\n"
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
			"assign out_data[" << node << "] = lut_" << node << "_ff;\n";
		"\n";

		os <<
			"\n"
			"\n";
	}

	os <<
		"endmodule\n";
	os << std::endl;
}


}