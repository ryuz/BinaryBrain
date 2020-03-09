// ---------------------------------------------------------------------------
//
//                                  Copyright (C) 2015-2018 by Ryuji Fuchikami
//                                      http://ryuz.my.coocan.jp/
// ---------------------------------------------------------------------------


`timescale 1ns / 1ps
`default_nettype none


module tb_mnist_lut_seg();
	localparam RATE = 1000.0/300.0;
	
	initial begin
		$dumpfile("tb_mnist_lut_seg.vcd");
		$dumpvars(3, tb_mnist_lut_seg);
//		$dumpvars(0, tb_mnist_lut_seg.i_video_mnist_color);
//		$dumpvars(3, tb_mnist_lut_seg.i_video_mnist);
//		$dumpvars(4, tb_mnist_lut_seg.i_video_mnist.i_video_mnist_core);
		
		
	#20000000
		$finish;
	end
	
	reg		reset = 1'b1;
	initial	#(RATE*100)	reset = 1'b0;
	
	reg		clk = 1'b1;
	always #(RATE/2.0)	clk = ~clk;
	
	wire	cke = 1'b1;
	
	
	localparam IMG_X_NUM = 640;
	localparam IMG_Y_NUM = 480;
//	localparam IMG_X_NUM = 640 / 4;
//	localparam IMG_Y_NUM = 480 / 4;
	
	
	localparam	DATA_WIDTH         = 8;
	localparam	NUM_CALSS          = 11;
	localparam	CHANNEL_WIDTH      = 1;
	
	localparam	IMG_Y_WIDTH        = 12;
	
	localparam	TUSER_WIDTH        = 1;
	localparam	S_TDATA_WIDTH      = 8;
	
	localparam	M_TNUMBER_WIDTH    = 4;
	localparam	M_TCOUNT_WIDTH     = 4;
	localparam	M_CLUSTERING_WIDTH = 11;
	
	localparam	WB_ADR_WIDTH       = 8;
	localparam	WB_DAT_WIDTH       = 32;
	localparam	WB_SEL_WIDTH       = (WB_DAT_WIDTH / 8);
	localparam	INIT_PARAM_TH      = 127;
	localparam	INIT_PARAM_INV     = 1'b0;
	
	
	
	
	// ----------------------------------
	//  dummy video
	// ----------------------------------
	
	wire	[0:0]	s_axi4s_tuser;
	wire			s_axi4s_tlast;
	wire	[31:0]	s_axi4s_tdata;
	wire			s_axi4s_tvalid;
	wire			s_axi4s_tready;
	
	jelly_axi4s_master_model
			#(
				.AXI4S_DATA_WIDTH	(24),
				.X_NUM				(IMG_X_NUM),
				.Y_NUM				(IMG_Y_NUM),
				.PPM_FILE			("mnist_test_640x480.ppm"),
		//		.PPM_FILE			("mnist_test_160x120.ppm"),
				.BUSY_RATE			(0), // (5),
				.RANDOM_SEED		(776),
				.INTERVAL			(1000)
			)
		i_axi4s_master_model
			(
				.aresetn			(~reset),
				.aclk				(clk),
				
				.m_axi4s_tuser		(s_axi4s_tuser),
				.m_axi4s_tlast		(s_axi4s_tlast),
				.m_axi4s_tdata		(s_axi4s_tdata[23:0]),
				.m_axi4s_tvalid		(s_axi4s_tvalid),
				.m_axi4s_tready		(s_axi4s_tready)
			);
	
	assign s_axi4s_tdata[31:24] = 0;
	
	
	
	// ----------------------------------
	//  image dump
	// ----------------------------------
	
	localparam FRAME_NUM = 1;
	
	integer		fp_img0;
	initial begin
		 fp_img0 = $fopen("src_img0.ppm", "w");
		 $fdisplay(fp_img0, "P3");
		 $fdisplay(fp_img0, "%d %d", IMG_X_NUM, IMG_Y_NUM*FRAME_NUM);
		 $fdisplay(fp_img0, "255");
	end
	
	always @(posedge clk) begin
		if ( !reset && s_axi4s_tvalid && s_axi4s_tready ) begin
			 $fdisplay(fp_img0, "%d %d %d", s_axi4s_tdata[0*8 +: 8], s_axi4s_tdata[1*8 +: 8], s_axi4s_tdata[2*8 +: 8]);
		end
	end
	
	
	// ----------------------------------
	//  MNIST
	// ----------------------------------
	
	wire	[TUSER_WIDTH-1:0]			m_axi4s_tuser;
	wire								m_axi4s_tlast;
	wire	[M_TNUMBER_WIDTH-1:0]		m_axi4s_tnumber;
	wire	[M_TCOUNT_WIDTH-1:0]		m_axi4s_tcount;
	wire	[M_CLUSTERING_WIDTH-1:0]	m_axi4s_tclustering;
	wire								m_axi4s_tvalid;
	wire								m_axi4s_tready;
	
	wire								s_wb_rst_i = reset;
	wire								s_wb_clk_i = clk;
	wire	[WB_ADR_WIDTH-1:0]			s_wb_adr_i;
	wire	[WB_DAT_WIDTH-1:0]			s_wb_dat_i;
	wire	[WB_DAT_WIDTH-1:0]			s_wb_dat_o;
	wire								s_wb_we_i;
	wire	[WB_SEL_WIDTH-1:0]			s_wb_sel_i;
	wire								s_wb_stb_i;
	wire								s_wb_ack_o;
	
	video_mnist_seg
			#(
				.DATA_WIDTH				(DATA_WIDTH),
				.NUM_CALSS				(NUM_CALSS),
				.CHANNEL_WIDTH			(CHANNEL_WIDTH),
				.IMG_Y_NUM				(IMG_Y_NUM),
				.IMG_Y_WIDTH			(IMG_Y_WIDTH),
				.TUSER_WIDTH			(TUSER_WIDTH),
				.S_TDATA_WIDTH			(S_TDATA_WIDTH),
				.WB_ADR_WIDTH			(WB_ADR_WIDTH),
				.WB_DAT_WIDTH			(WB_DAT_WIDTH),
				.WB_SEL_WIDTH			(WB_SEL_WIDTH),
				.INIT_PARAM_TH			(INIT_PARAM_TH),
				.INIT_PARAM_INV			(INIT_PARAM_INV)
			)
		i_video_mnist_seg
			(
				.aresetn				(~reset),
				.aclk					(clk),
				
				.s_axi4s_tuser			(s_axi4s_tuser),
				.s_axi4s_tlast			(s_axi4s_tlast),
				.s_axi4s_tdata			(s_axi4s_tdata[7:0]),
				.s_axi4s_tvalid			(s_axi4s_tvalid),
				.s_axi4s_tready			(s_axi4s_tready),
				
				.m_axi4s_tuser			(m_axi4s_tuser),
				.m_axi4s_tlast			(m_axi4s_tlast),
				.m_axi4s_tclustering	(m_axi4s_tclustering),
				.m_axi4s_tnumber		(m_axi4s_tnumber),
				.m_axi4s_tcount			(m_axi4s_tcount),
				.m_axi4s_tvalid			(m_axi4s_tvalid),
				.m_axi4s_tready			(m_axi4s_tready),
				
				.s_wb_rst_i				(s_wb_rst_i),
				.s_wb_clk_i				(s_wb_clk_i),
				.s_wb_adr_i				(s_wb_adr_i),
				.s_wb_dat_i				(s_wb_dat_i),
				.s_wb_dat_o				(s_wb_dat_o),
				.s_wb_we_i				(s_wb_we_i),
				.s_wb_sel_i				(s_wb_sel_i),
				.s_wb_stb_i				(s_wb_stb_i),
				.s_wb_ack_o				(s_wb_ack_o)
			);
	
	
	wire	[0:0]				axi4s_color_tuser;
	wire						axi4s_color_tlast;
	wire	[31:0]				axi4s_color_tdata;
	wire						axi4s_color_tvalid;
	wire						axi4s_color_tready;
	
	video_mnist_seg_color
			#(
				.DATA_WIDTH			(DATA_WIDTH),
				.TUSER_WIDTH		(1),
				.INIT_PARAM_MODE	(3'b111),
				.INIT_PARAM_TH		(1)
			)
		i_video_mnist_seg_color
			(
				.aresetn			(~reset),
				.aclk				(clk),
				
				.s_axi4s_tuser		(m_axi4s_tuser),
				.s_axi4s_tlast		(m_axi4s_tlast),
				.s_axi4s_tnumber	(m_axi4s_tnumber),
				.s_axi4s_tcount		(m_axi4s_tcount),
				.s_axi4s_tdata		(32'h00202020),		// (m_axi4s_tdata),
				.s_axi4s_tbinary	(1'b0),				// (m_axi4s_tbinary),
				.s_axi4s_tdetection (1'b1),
				.s_axi4s_tvalid		(m_axi4s_tvalid),
				.s_axi4s_tready		(m_axi4s_tready),
				
				.m_axi4s_tuser		(axi4s_color_tuser),
				.m_axi4s_tlast		(axi4s_color_tlast),
				.m_axi4s_tdata		(axi4s_color_tdata),
				.m_axi4s_tvalid		(axi4s_color_tvalid),
				.m_axi4s_tready		(axi4s_color_tready),
				
				.s_wb_rst_i			(s_wb_rst_i),
				.s_wb_clk_i			(s_wb_clk_i),
				.s_wb_adr_i			(8'd0),
				.s_wb_dat_i			(32'd0),
				.s_wb_dat_o			(),
				.s_wb_we_i			(1'b0),
				.s_wb_sel_i			(4'd0),
				.s_wb_stb_i			(1'b0),
				.s_wb_ack_o			()
		);
	
	
	
	// 出力結果を保存 (サイズは 1/28 )
	jelly_axi4s_slave_model
			#(
				.COMPONENT_NUM		(3),
				.DATA_WIDTH			(DATA_WIDTH),
				.INIT_FRAME_NUM		(0),
				.FRAME_WIDTH		(32),
				.X_WIDTH			(32),
				.Y_WIDTH			(32),
				.FILE_NAME			("col_%1d.ppm"),
				.MAX_PATH			(64),
				.BUSY_RATE			(0),
				.RANDOM_SEED		(1234)
			)
		jelly_axi4s_slave_model_col
			(
				.aresetn			(~reset),
				.aclk				(clk),
				.aclken				(1'b1),
				
				.param_width		(IMG_X_NUM),
				.param_height		(IMG_Y_NUM),
				
				.s_axi4s_tuser		(axi4s_color_tuser),
				.s_axi4s_tlast		(axi4s_color_tlast),
				.s_axi4s_tdata		({axi4s_color_tdata[7:0], axi4s_color_tdata[15:8], axi4s_color_tdata[23:16]}),
				.s_axi4s_tvalid		(axi4s_color_tvalid),
				.s_axi4s_tready		(axi4s_color_tready)
			);
	
	
	// 出力フレームカウント
	integer	output_frame = 0;
	always @(posedge clk) begin
		if ( !reset ) begin
			if ( axi4s_color_tvalid && axi4s_color_tready && axi4s_color_tuser[0] ) begin
				output_frame <= output_frame + 1;
			end
			
			if ( output_frame >= 2 ) begin
				$finish();
			end
		end
	end
	
	
	/*
	jelly_axi4s_slave_model
			#(
				.COMPONENT_NUM		(3),
				.DATA_WIDTH			(DATA_WIDTH),
				.INIT_FRAME_NUM		(0),
				.FRAME_WIDTH		(32),
				.X_WIDTH			(32),
				.Y_WIDTH			(32),
				.FILE_NAME			("rgb_%1d.ppm"),
				.MAX_PATH			(64),
				.BUSY_RATE			(0),
				.RANDOM_SEED		(1234)
			)
		jelly_axi4s_slave_model_rgb
			(
				.aresetn			(~reset),
				.aclk				(clk),
				.aclken				(1'b1),
				
				.param_width		(IMG_X_NUM),
				.param_height		(IMG_Y_NUM),
				
				.s_axi4s_tuser		(m_axi4s_tuser),
				.s_axi4s_tlast		(m_axi4s_tlast),
				.s_axi4s_tdata		(m_axi4s_tdata[23:0]),
				.s_axi4s_tvalid		(m_axi4s_tvalid & m_axi4s_tready),
				.s_axi4s_tready		()
			);
	*/
	
	
	// ----------------------------------
	//
	// ----------------------------------
	
	/*
	wire				img_blk_cke   = i_video_mnist.i_img_mnist.i_img_mnist_core.cke;
	wire				img_blk_de    = i_video_mnist.i_img_mnist.i_img_mnist_core.img_blk_de;
	wire	[28*28-1:0]	img_blk_data  = i_video_mnist.i_img_mnist.i_img_mnist_core.img_blk_data;
	wire				img_blk_valid = i_video_mnist.i_img_mnist.i_img_mnist_core.img_blk_valid;
	
	integer					blk_frame = 0;
	reg		[256*8-1:0]		blk_filename;
	integer					blk_fp = 0;
	integer					blk_i;
	
	always @(posedge clk) begin
		if ( !reset && img_blk_cke && img_blk_valid && img_blk_de ) begin
			$sformat(blk_filename, "blk/blk_%04d.pgm", blk_frame);
			blk_fp = $fopen(blk_filename, "w");
			$fdisplay(blk_fp, "P2");
			$fdisplay(blk_fp, "28 28");
			$fdisplay(blk_fp, "1");
			for ( blk_i = 0; blk_i < 28*28; blk_i = blk_i+1 ) begin
				$fdisplay(blk_fp, "%d", img_blk_data[blk_i]);
			end
			$fclose(blk_fp);
			blk_frame = blk_frame + 1;
		end
	end
	*/
	
	
	// ----------------------------------
	//  WISHBONE master
	// ----------------------------------
	
	
	wire							wb_rst_i = s_wb_rst_i;
	wire							wb_clk_i = s_wb_clk_i;
	reg		[WB_ADR_WIDTH-1:0]		wb_adr_o;
	wire	[WB_DAT_WIDTH-1:0]		wb_dat_i = s_wb_dat_o;
	reg		[WB_DAT_WIDTH-1:0]		wb_dat_o;
	reg								wb_we_o;
	reg		[WB_SEL_WIDTH-1:0]		wb_sel_o;
	reg								wb_stb_o = 0;
	wire							wb_ack_i = s_wb_ack_o;
	
	initial begin
		force s_wb_adr_i = wb_adr_o;
		force s_wb_dat_i = wb_dat_o;
		force s_wb_we_i  = wb_we_o;
		force s_wb_sel_i = wb_sel_o;
		force s_wb_stb_i = wb_stb_o;
	end
	
	
	reg		[WB_DAT_WIDTH-1:0]		reg_wb_dat;
	reg								reg_wb_ack;
	always @(posedge wb_clk_i) begin
		if ( ~wb_we_o & wb_stb_o & wb_ack_i ) begin
			reg_wb_dat <= wb_dat_i;
		end
		reg_wb_ack <= wb_ack_i;
	end
	
	
	task wb_write(
				input [31:0]	adr,
				input [31:0]	dat,
				input [3:0]		sel
			);
	begin
		$display("WISHBONE_WRITE(adr:%h dat:%h sel:%b)", adr, dat, sel);
		@(negedge wb_clk_i);
			wb_adr_o = (adr >> 2);
			wb_dat_o = dat;
			wb_sel_o = sel;
			wb_we_o  = 1'b1;
			wb_stb_o = 1'b1;
		@(negedge wb_clk_i);
			while ( reg_wb_ack == 1'b0 ) begin
				@(negedge wb_clk_i);
			end
			wb_adr_o = {WB_ADR_WIDTH{1'bx}};
			wb_dat_o = {WB_DAT_WIDTH{1'bx}};
			wb_sel_o = {WB_SEL_WIDTH{1'bx}};
			wb_we_o  = 1'bx;
			wb_stb_o = 1'b0;
	end
	endtask
	
	task wb_read(
				input [31:0]	adr
			);
	begin
		@(negedge wb_clk_i);
			wb_adr_o = (adr >> 2);
			wb_dat_o = {WB_DAT_WIDTH{1'bx}};
			wb_sel_o = {WB_SEL_WIDTH{1'b1}};
			wb_we_o  = 1'b0;
			wb_stb_o = 1'b1;
		@(negedge wb_clk_i);
			while ( reg_wb_ack == 1'b0 ) begin
				@(negedge wb_clk_i);
			end
			wb_adr_o = {WB_ADR_WIDTH{1'bx}};
			wb_dat_o = {WB_DAT_WIDTH{1'bx}};
			wb_sel_o = {WB_SEL_WIDTH{1'bx}};
			wb_we_o  = 1'bx;
			wb_stb_o = 1'b0;
			$display("WISHBONE_READ(adr:%h dat:%h)", adr, reg_wb_dat);
	end
	endtask
	
	
	initial begin
	@(negedge wb_rst_i);
	#10000;
//		$display("start");
//		wb_write(32'h00010010, 32'h00, 4'b1111);
	end
	
	
endmodule


`default_nettype wire


// end of file
