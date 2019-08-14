// ---------------------------------------------------------------------------
//
//                                 Copyright (C) 2015-2018 by Ryuji Fuchikami
//                                 http://ryuz.my.coocan.jp/
// ---------------------------------------------------------------------------


`timescale 1ns / 1ps
`default_nettype none


module tb_mnist_lut_simple();
	localparam RATE = 1000.0/300.0;
	
	initial begin
		$dumpfile("tb_mnist_lut_simple.vcd");
		$dumpvars(2, tb_mnist_lut_simple);

	#1000000
		$finish();
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
	localparam	CLASS_NUM    = 10;
	localparam	CHANNEL_NUM  = 7;		// チャネル方向(空間的に)多重
	localparam	OUTPUT_WIDTH = CLASS_NUM * CHANNEL_NUM;
	
	reg		[USER_WIDTH+INPUT_WIDTH-1:0]	mem		[0:DATA_SIZE-1];
	initial begin
		$readmemb(FILE_NAME, mem);
	end
	
	
	integer									index = 0;
	wire									in_last = (index == DATA_SIZE-1);
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
		end
	end
	
	
	wire								out_last;
	wire		[USER_WIDTH-1:0]		out_user;
	wire		[OUTPUT_WIDTH-1:0]		out_data;
	wire								out_valid;
	
	MnistSparseLutSimple
			#(
				.USER_WIDTH		(1+USER_WIDTH)
			)
		i_MnistSparseLutSimple
			(
				.reset			(reset),
				.clk			(clk),
				.cke			(cke),
				
				.in_user		({in_last, in_user}),
				.in_data		(in_data),
				.in_valid		(in_valid),
				
				.out_user		({out_last, out_user}),
				.out_data		(out_data),
				.out_valid		(out_valid)
			);
	
	
	// デバッグ用に追加
	wire	[27:0]	dbg_in_data_027_000 = in_data[027:000];
	wire	[27:0]	dbg_in_data_055_028 = in_data[055:028];
	wire	[27:0]	dbg_in_data_083_056 = in_data[083:056];
	wire	[27:0]	dbg_in_data_111_084 = in_data[111:084];
	wire	[27:0]	dbg_in_data_139_112 = in_data[139:112];
	wire	[27:0]	dbg_in_data_167_140 = in_data[167:140];
	wire	[27:0]	dbg_in_data_195_168 = in_data[195:168];
	wire	[27:0]	dbg_in_data_223_196 = in_data[223:196];
	wire	[27:0]	dbg_in_data_251_224 = in_data[251:224];
	wire	[27:0]	dbg_in_data_279_252 = in_data[279:252];
	wire	[27:0]	dbg_in_data_307_280 = in_data[307:280];
	wire	[27:0]	dbg_in_data_335_308 = in_data[335:308];
	wire	[27:0]	dbg_in_data_363_336 = in_data[363:336];
	wire	[27:0]	dbg_in_data_391_364 = in_data[391:364];
	wire	[27:0]	dbg_in_data_419_392 = in_data[419:392];
	wire	[27:0]	dbg_in_data_447_420 = in_data[447:420];
	wire	[27:0]	dbg_in_data_475_448 = in_data[475:448];
	wire	[27:0]	dbg_in_data_503_476 = in_data[503:476];
	wire	[27:0]	dbg_in_data_531_504 = in_data[531:504];
	wire	[27:0]	dbg_in_data_559_532 = in_data[559:532];
	wire	[27:0]	dbg_in_data_587_560 = in_data[587:560];
	wire	[27:0]	dbg_in_data_615_588 = in_data[615:588];
	wire	[27:0]	dbg_in_data_643_616 = in_data[643:616];
	wire	[27:0]	dbg_in_data_671_644 = in_data[671:644];
	wire	[27:0]	dbg_in_data_699_672 = in_data[699:672];
	wire	[27:0]	dbg_in_data_727_700 = in_data[727:700];
	wire	[27:0]	dbg_in_data_755_728 = in_data[755:728];
	wire	[27:0]	dbg_in_data_783_756 = in_data[783:756];
	
	wire	[2:0]	dbg_out_sum0 = out_data[0] + out_data[10] + out_data[20] + out_data[30] + out_data[40] + out_data[50] + out_data[60];
	wire	[2:0]	dbg_out_sum1 = out_data[1] + out_data[11] + out_data[21] + out_data[31] + out_data[41] + out_data[51] + out_data[61];
	wire	[2:0]	dbg_out_sum2 = out_data[2] + out_data[12] + out_data[22] + out_data[32] + out_data[42] + out_data[52] + out_data[62];
	wire	[2:0]	dbg_out_sum3 = out_data[3] + out_data[13] + out_data[23] + out_data[33] + out_data[43] + out_data[53] + out_data[63];
	wire	[2:0]	dbg_out_sum4 = out_data[4] + out_data[14] + out_data[24] + out_data[34] + out_data[44] + out_data[54] + out_data[64];
	wire	[2:0]	dbg_out_sum5 = out_data[5] + out_data[15] + out_data[25] + out_data[35] + out_data[45] + out_data[55] + out_data[65];
	wire	[2:0]	dbg_out_sum6 = out_data[6] + out_data[16] + out_data[26] + out_data[36] + out_data[46] + out_data[56] + out_data[66];
	wire	[2:0]	dbg_out_sum7 = out_data[7] + out_data[17] + out_data[27] + out_data[37] + out_data[47] + out_data[57] + out_data[67];
	wire	[2:0]	dbg_out_sum8 = out_data[8] + out_data[18] + out_data[28] + out_data[38] + out_data[48] + out_data[58] + out_data[68];
	wire	[2:0]	dbg_out_sum9 = out_data[9] + out_data[19] + out_data[29] + out_data[39] + out_data[49] + out_data[59] + out_data[69];
	
	
	
	// sum
	integer							i, j;
	integer							tmp;
	reg								sum_last;
	reg		[CLASS_NUM*32-1:0]		sum_data;
	reg		[USER_WIDTH-1:0]		sum_user;
	reg								sum_valid;
	always @(posedge clk) begin
		if ( reset ) begin
			sum_last  <= 0;
			sum_user  <= 0;
			sum_valid <= 1'b0;
		end
		else if  ( cke ) begin
			for ( i = 0; i < CLASS_NUM; i = i+1 ) begin
				tmp = 0;
				for ( j = 0; j < CHANNEL_NUM; j = j+1 ) begin
					tmp = tmp + out_data[j*CLASS_NUM + i];
				end
				sum_data[i*32 +: 32] <= tmp;
			end
			
			sum_last    <= out_last;
			sum_user    <= out_user;
			sum_valid   <= out_valid;
		end
	end
	
	integer		max_index;
	integer		max_value;
	always @* begin
		max_index = -1;
		max_value = 0;
		for ( i = 0; i < CLASS_NUM; i = i+1 ) begin
			if ( sum_data[i*32 +: 32] > max_value ) begin
				max_index = i;
				max_value = sum_data[i*32 +: 32];
			end
		end
	end
	
	wire match = (max_index == sum_user);
	
	// 結果カウント
	integer		out_data_counter = 0;
	integer		out_ok_counter   = 0;
	always @(posedge clk) begin
		if ( reset ) begin
			out_data_counter = 0;
			out_ok_counter   = 0;
		end
		else begin
			if ( sum_valid ) begin
				out_data_counter = out_data_counter + 1;
				out_ok_counter   = out_ok_counter   + match;
				if ( sum_last ) begin
					$display("accuracy = %d/%d", out_ok_counter, out_data_counter);
					$finish;
				end
			end
		end
	end
	
	
endmodule


`default_nettype wire


// end of file
