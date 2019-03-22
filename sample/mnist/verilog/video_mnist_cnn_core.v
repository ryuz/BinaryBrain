// ---------------------------------------------------------------------------
//  Jelly  -- the soft-core processor system
//   math
//
//                                 Copyright (C) 2008-2018 by Ryuji Fuchikami
//                                 http://ryuz.my.coocan.jp/
// ---------------------------------------------------------------------------



`timescale 1ns / 1ps
`default_nettype none


module video_mnist_cnn_core
		#(
			parameter	TUSER_WIDTH    = 1,
			
			parameter	MAX_X_NUM      = 1024,
			parameter	IMG_Y_NUM      = 480,
			parameter	IMG_X_WIDTH    = 11,
			parameter	IMG_Y_WIDTH    = 10,
			parameter	BLANK_Y_WIDTH  = 8,
			
			parameter	INIT_Y_NUM     = IMG_Y_NUM,
			
			parameter	S_TDATA_WIDTH  = 1,
			parameter	M_TDATA_WIDTH  = 80,

			parameter	RAM_TYPE       = "block",
			parameter	FIFO_PTR_WIDTH = 9,
			parameter	FIFO_RAM_TYPE  = "block",
			parameter	IMG_CKE_BUFG   = 0,
			
			
			parameter	DEVICE         = "rtl"
		)
		(
			input	wire						aresetn,
			input	wire						aclk,
			
			input	wire	[BLANK_Y_WIDTH-1:0]	param_blank_num,
			
			input	wire	[TUSER_WIDTH-1:0]	s_axi4s_tuser,
			input	wire						s_axi4s_tlast,
			input	wire	[S_TDATA_WIDTH-1:0]	s_axi4s_tdata,
			input	wire						s_axi4s_tvalid,
			output	wire						s_axi4s_tready,
			
			output	wire	[TUSER_WIDTH-1:0]	m_axi4s_tuser,
			output	wire						m_axi4s_tlast,
			output	wire	[M_TDATA_WIDTH-1:0]	m_axi4s_tdata,
			output	wire						m_axi4s_tvalid,
			input	wire						m_axi4s_tready
		);
	
	localparam	L0_TDATA_WIDTH = 32;
	localparam	L1_TDATA_WIDTH = 32;
	localparam	L2_TDATA_WIDTH = M_TDATA_WIDTH;
	
	
	// L0
	wire	[TUSER_WIDTH-1:0]		axi4s_l0_tuser;
	wire							axi4s_l0_tlast;
	wire	[L0_TDATA_WIDTH-1:0]	axi4s_l0_tdata;
	wire							axi4s_l0_tvalid;
	wire							axi4s_l0_tready;
	
	MnistSimpleLutCnnCnv0
			#(
				.TUSER_WIDTH		(TUSER_WIDTH),
				.IMG_X_WIDTH		(IMG_X_WIDTH),
				.IMG_Y_WIDTH		(IMG_Y_WIDTH),
				.IMG_Y_NUM			(IMG_Y_NUM),
				.MAX_X_NUM			(MAX_X_NUM),
				.BLANK_Y_WIDTH  	(BLANK_Y_WIDTH),
				.INIT_Y_NUM			(INIT_Y_NUM),
				.FIFO_PTR_WIDTH		(FIFO_PTR_WIDTH),
				.FIFO_RAM_TYPE		(FIFO_RAM_TYPE),
				.IMG_CKE_BUFG		(IMG_CKE_BUFG),
				.DEVICE         	(DEVICE)
			)
		i_MnistSimpleLutCnnCnv0
			(
				.reset				(~aresetn),
				.clk				(aclk),
				
				.param_blank_num	(param_blank_num),
				
				.s_axi4s_tuser		(s_axi4s_tuser),
				.s_axi4s_tlast		(s_axi4s_tlast),
				.s_axi4s_tdata		(s_axi4s_tdata),
				.s_axi4s_tvalid		(s_axi4s_tvalid),
				.s_axi4s_tready		(s_axi4s_tready),
				
				.m_axi4s_tuser		(axi4s_l0_tuser),
				.m_axi4s_tlast		(axi4s_l0_tlast),
				.m_axi4s_tdata		(axi4s_l0_tdata),
				.m_axi4s_tvalid		(axi4s_l0_tvalid),
				.m_axi4s_tready		(axi4s_l0_tready)
			);
	
	
	// L1
	wire	[TUSER_WIDTH-1:0]		axi4s_l1_tuser;
	wire							axi4s_l1_tlast;
	wire	[L1_TDATA_WIDTH-1:0]	axi4s_l1_tdata;
	wire							axi4s_l1_tvalid;
	wire							axi4s_l1_tready;
	
	MnistSimpleLutCnnCnv1
			#(
				.TUSER_WIDTH		(TUSER_WIDTH),
				.IMG_X_WIDTH		(IMG_X_WIDTH),
				.IMG_Y_WIDTH		(IMG_Y_WIDTH),
				.IMG_Y_NUM			(IMG_Y_NUM/2),
				.MAX_X_NUM			(MAX_X_NUM/2),
				.BLANK_Y_WIDTH  	(BLANK_Y_WIDTH),
				.INIT_Y_NUM			(INIT_Y_NUM/2),
				.FIFO_PTR_WIDTH		(FIFO_PTR_WIDTH),
				.FIFO_RAM_TYPE		(FIFO_RAM_TYPE),
				.IMG_CKE_BUFG		(IMG_CKE_BUFG),
				.DEVICE         	(DEVICE)
			)
		i_MnistSimpleLutCnnCnv1
			(
				.reset				(~aresetn),
				.clk				(aclk),
				
				.param_blank_num	(param_blank_num),
				
				.s_axi4s_tuser		(axi4s_l0_tuser),
				.s_axi4s_tlast		(axi4s_l0_tlast),
				.s_axi4s_tdata		(axi4s_l0_tdata),
				.s_axi4s_tvalid		(axi4s_l0_tvalid),
				.s_axi4s_tready		(axi4s_l0_tready),
				
				.m_axi4s_tuser		(axi4s_l1_tuser),
				.m_axi4s_tlast		(axi4s_l1_tlast),
				.m_axi4s_tdata		(axi4s_l1_tdata),
				.m_axi4s_tvalid		(axi4s_l1_tvalid),
				.m_axi4s_tready		(axi4s_l1_tready)
			);
	
	
	// L2
	MnistSimpleLutCnnCnv2
			#(
				.TUSER_WIDTH		(TUSER_WIDTH),
				.IMG_X_WIDTH		(IMG_X_WIDTH),
				.IMG_Y_WIDTH		(IMG_Y_WIDTH),
				.IMG_Y_NUM			(IMG_Y_NUM/4),
				.MAX_X_NUM			(MAX_X_NUM/4),
				.BLANK_Y_WIDTH  	(BLANK_Y_WIDTH),
				.INIT_Y_NUM			(INIT_Y_NUM/4),
				.FIFO_PTR_WIDTH		(FIFO_PTR_WIDTH),
				.FIFO_RAM_TYPE		(FIFO_RAM_TYPE),
				.IMG_CKE_BUFG		(IMG_CKE_BUFG),
				.DEVICE         	(DEVICE)
			)
		i_MnistSimpleLutCnnCnv2
			(
				.reset				(~aresetn),
				.clk				(aclk),
				
				.param_blank_num	(param_blank_num),
				
				.s_axi4s_tuser		(axi4s_l1_tuser),
				.s_axi4s_tlast		(axi4s_l1_tlast),
				.s_axi4s_tdata		(axi4s_l1_tdata),
				.s_axi4s_tvalid		(axi4s_l1_tvalid),
				.s_axi4s_tready		(axi4s_l1_tready),
				
				.m_axi4s_tuser		(m_axi4s_tuser),
				.m_axi4s_tlast		(m_axi4s_tlast),
				.m_axi4s_tdata		(m_axi4s_tdata),
				.m_axi4s_tvalid		(m_axi4s_tvalid),
				.m_axi4s_tready		(m_axi4s_tready)
			);
	
	
endmodule



`default_nettype wire



// end of file
