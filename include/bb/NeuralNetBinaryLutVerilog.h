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

#include "bb/NeuralNetBinaryLut.h"


namespace bb {

// Verilog 出力
template <bool bitwise=false, typename T = float, typename INDEX = size_t>
void NeuralNetBinaryLutVerilog(std::ostream& os, NeuralNetBinaryLut<bitwise, T, INDEX>& lut, std::string module_name)
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


}