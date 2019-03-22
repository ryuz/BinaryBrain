// ---------------------------------------------------------------------------
//
//                                  Copyright (C) 2015-2018 by Ryuji Fuchikami
//                                      http://ryuz.my.coocan.jp/
// ---------------------------------------------------------------------------


`timescale 1ns / 1ps
`default_nettype none


module tb_mnist_lut_net();
	localparam RATE = 1000.0/300.0;
	
	initial begin
		$dumpfile("tb_mnist_lut_net.vcd");
		$dumpvars(2, tb_mnist_lut_net);
		
	#20000000
		$finish;
	end
	
	reg		reset = 1'b1;
	initial	#(RATE*100)	reset = 1'b0;
	
	reg		clk = 1'b1;
	always #(RATE/2.0)	clk = ~clk;
	
	wire	cke = 1'b1;
	
	
	
	localparam	FILE_NAME    = "mnist_test.txt";
	localparam	DATA_SIZE    = 10000;
	localparam	USER_WIDTH   = 8;
	localparam	INPUT_WIDTH  = 28*28;
	localparam	OUTPUT_WIDTH = 10;
	
	reg		[USER_WIDTH+INPUT_WIDTH-1:0]	mem		[0:DATA_SIZE-1];
	initial begin
		$readmemb(FILE_NAME, mem);
	end
	
	
	integer									index = 0;
	wire		[USER_WIDTH-1:0]			in_user;
	wire		[INPUT_WIDTH-1:0]			in_data;
	reg										in_valid = 0;
	
	assign {in_user, in_data} = in_valid ? mem[index] : {(USER_WIDTH+INPUT_WIDTH){1'bx}};
	
	always @(posedge clk) begin
		if ( reset ) begin
			index    <= 0;
			in_valid <= 1'b0;
		end
		else begin
			index    <= index + in_valid;
			in_valid <= 1'b1;
			
			if ( index == DATA_SIZE-1 ) begin
//				index <= 0;
				$finish();
			end
		end
	end
	
	
	wire		[USER_WIDTH-1:0]		out_user;
	wire		[OUTPUT_WIDTH-1:0]		out_data;
	wire								out_valid;
	
	MnistSimpleLutMlp
			#(
				.USER_WIDTH		(USER_WIDTH)
			)
		i_mnist_lut_net
			(
				.reset			(reset),
				.clk			(clk),
				.cke			(cke),
				
				.in_user		(in_user),
				.in_data		(in_data),
				.in_valid		(in_valid),
				
				.out_user		(out_user),
				.out_data		(out_data),
				.out_valid		(out_valid)
			);
	
	// 期待値
	wire	[OUTPUT_WIDTH-1:0]	out_expect = (1 << out_user);
	
	wire match = out_valid && (out_data == out_expect);
	
	
endmodule


`default_nettype wire


// end of file
