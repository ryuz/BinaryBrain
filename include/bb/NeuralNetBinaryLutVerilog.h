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

#include "bb/NeuralNetBinaryLut.h"


namespace bb {

// Verilog 出力
template <typename T = float>
void NeuralNetBinaryLutVerilog(std::ostream& os, NeuralNetBinaryLut<T>& lut, std::string module_name)
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
void NeuralNetMultilayerBinaryLutVerilog(std::ostream& os, std::vector< NeuralNetBinaryLut<T>* > layers, std::string module_name)
{
	std::vector<std::string> sub_modle_name;

	auto first_layer = layers[0];
	auto last_layer = layers[layers.size() - 1];

	// サブモジュール名生成
	for (int i = 0; i < (int)layers.size(); ++i) {
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

	for (int i = 0; i < (int)layers.size(); ++i) {
		os
			<< "reg   [USER_BITS-1:0]  layer" << i << "_user;\n"
			<< "wire  [" << std::setw(9) << layers[i]->GetOutputNodeSize() << "-1:0]  layer" << i << "_data;\n"
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
		<< "assign out_data  = layer" << (layers.size() - 1) << "_data;\n"
		<< "assign out_user  = layer" << (layers.size() - 1) << "_user;\n"
		<< "assign out_valid = layer" << (layers.size() - 1) << "_valid;\n"
		<< "\n"
		<< "endmodule\n"
		<< "\n\n";
	

	// サブモジュール出力
	for (int i = 0; i < (int)layers.size(); ++i) {
		NeuralNetBinaryLutVerilog(os, *layers[i], sub_modle_name[i]);
	}
}



}