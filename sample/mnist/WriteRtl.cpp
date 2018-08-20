#include <iostream>
#include <sstream>
#include "BinaryNetData.h"



void WriteRtl(std::ostream& os, BinaryNetData& bnd)
{
	int layer_num  = (int)bnd.layer.size();
	if (layer_num <= 0) { return; }
	int input_num  = (int)bnd.input_num;
	int output_num = (int)bnd.layer[layer_num-1].node.size();

	os <<
		"\n"
		"\n"
		"module binary_net(\n"
		"            input  wire         reset,\n"
		"            input  wire         clk,\n"
		"            input  wire         cke,\n"
		"            \n"
		"            input  wire [" << (input_num - 1) << ":0]  in_data,\n"
		"            output wire [" << (output_num - 1) << ":0]  out_data\n"
		"        );\n"
		"\n";

	for (int layer = 0; layer < layer_num; layer++) {
		int node_num = (int)bnd.layer[layer].node.size();
	
		os <<
			"\n"
			"// -------------------------------------------\n"
			"//  Layer" << layer << "\n"
			"// -------------------------------------------\n"
			"\n";

		os << "wire  [" << (node_num - 1) << ":0]    layer" << layer << "_data;\n";

		for (int node = 0; node < node_num; node++) {
			os <<
				"\n"
				"// LUT " << layer << "-" << node << "\n";
			
			auto& lut = bnd.layer[layer].node[node];

			std::stringstream  ssId;
			std::stringstream  ssI;
			std::stringstream  ssO;
			std::stringstream  ssF;

			ssId << layer << "_" << node;

			if (layer == 0) {
				ssI << "in_data";
			}
			else {
				ssI << "layer" << (layer - 1) << "_data";
			}

			ssO << "lut_" << ssId.str() << "_out";
			ssF << "lut_" << ssId.str() << "_ff";

			os << "wire  " << ssO.str() << ";\n";

			os <<
				"LUT6\n"
				"        #(\n"
				"            .INIT(64'b";

			
			for (auto& it = lut.lut.rbegin(); it != lut.lut.rend(); ++it) {
				os << *it ? "1" : "0";
			}
			os << ")\n";

			os <<
				"        )\n"
				"    i_lut6_" << layer << "_" << node << "\n"
				"        (\n"
				"            .O  (" << ssO.str() << "),\n"
				"            .I0 (" << ssI.str() << "[" << lut.connect[0] << "]),\n"
				"            .I1 (" << ssI.str() << "[" << lut.connect[1] << "]),\n"
				"            .I2 (" << ssI.str() << "[" << lut.connect[2] << "]),\n"
				"            .I3 (" << ssI.str() << "[" << lut.connect[3] << "]),\n"
				"            .I4 (" << ssI.str() << "[" << lut.connect[4] << "]),\n"
				"            .I5 (" << ssI.str() << "[" << lut.connect[5] << "])\n"
				"        );\n"
				"\n";

			os <<
				"reg   lut_" << ssId.str() << "_ff;\n"
				"always @(posedge clk) begin\n"
				"    if ( reset ) begin\n"
				"        " << ssF.str() << " <= 1'b0;\n"
				"    end\n"
				"    else if ( cke ) begin\n"
				"        " << ssF.str() << " <= " << ssO.str() << ";\n"
				"    end\n"
				"end\n"
				"assign layer" << layer << "_data[" << node << "] = " << ssF.str() << ";\n";
				"\n";

			os <<
				"\n"
				"\n";
		}
	}

	// output
	{
		os <<
			"\n"
			"// -------------------------------------------\n"
			"//  Output\n"
			"// -------------------------------------------\n"
			"\n";

		int layer = layer_num - 1;
		for (int node = 0; node < output_num; node++) {
			os <<
				"assign out_data[" << node << "] = layer" << layer << "_data[" << node << "];\n";
		}
		os << "\n\n";
	}

	os <<
		"endmodule\n";
	os << std::endl;
}
