

`timescale 1ns / 1ps
`default_nettype none



module bb_lut
		#(
			parameter					DEVICE = "RTL",
			parameter					N      = 6,
			parameter	[(1<<N)-1:0]	INIT   = {(1<<N){1'b0}}
		)
		(
			input	wire	[N-1:0]		in_data,
			output	wire				out_data
		);
	
	generate
	if ( DEVICE == "7SERIES" && N == 6 ) begin : xilinx_7series
		LUT6
				#(
					.INIT	(INIT)
				)
			i_lut6
				(
					.O		(out_data),
					.I0		(in_data[0]),
					.I1		(in_data[1]),
					.I2		(in_data[2]),
					.I3		(in_data[3]),
					.I4		(in_data[4]),
					.I5		(in_data[5])
				);
	end
	else begin : rtl
		wire	[(1<<N)-1:0]	lut_table = INIT;
		assign out_data = lut_table[in_data];
	end
	endgenerate
	
	
endmodule



`default_nettype wire


