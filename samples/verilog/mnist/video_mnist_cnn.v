// ---------------------------------------------------------------------------
//  MNIST sample
//
//                                 Copyright (C) 2008-2021 by Ryuji Fuchikami
// ---------------------------------------------------------------------------



`timescale 1ns / 1ps
`default_nettype none


module video_mnist_cnn
        #(
            parameter   DATA_WIDTH         = 8,
            parameter   MAX_X_NUM          = 1024,
            parameter   RAM_TYPE           = "block",
            
            parameter   NUM_CALSS          = 10,
            parameter   CHANNEL_WIDTH      = 7,
            
            parameter   IMG_Y_NUM          = 480,
            parameter   IMG_Y_WIDTH        = 12,
            
            parameter   TUSER_WIDTH        = 1,
            parameter   S_TDATA_WIDTH      = DATA_WIDTH,
            parameter   M_TNUMBER_WIDTH    = 4,
            parameter   M_TCOUNT_WIDTH     = 4,
            parameter   M_CLUSTERING_WIDTH = NUM_CALSS * CHANNEL_WIDTH,
            
            parameter   DEVICE             = "rtl",
            
            parameter   WB_ADR_WIDTH       = 8,
            parameter   WB_DAT_WIDTH       = 32,
            parameter   WB_SEL_WIDTH       = (WB_DAT_WIDTH / 8),
            parameter   INIT_PARAM_TH      = 127,
            parameter   INIT_PARAM_INV     = 1'b0
        )
        (
            input   wire                                aresetn,
            input   wire                                aclk,
            
            input   wire    [TUSER_WIDTH-1:0]           s_axi4s_tuser,
            input   wire                                s_axi4s_tlast,
            input   wire    [S_TDATA_WIDTH-1:0]         s_axi4s_tdata,
            input   wire                                s_axi4s_tvalid,
            output  wire                                s_axi4s_tready,
            
            output  wire    [TUSER_WIDTH-1:0]           m_axi4s_tuser,
            output  wire                                m_axi4s_tlast,
            output  wire    [M_TNUMBER_WIDTH-1:0]       m_axi4s_tnumber,
            output  wire    [M_TCOUNT_WIDTH-1:0]        m_axi4s_tcount,
            output  wire    [M_CLUSTERING_WIDTH-1:0]    m_axi4s_tclustering,
            output  wire                                m_axi4s_tvalid,
            input   wire                                m_axi4s_tready,
            
            input   wire                                s_wb_rst_i,
            input   wire                                s_wb_clk_i,
            input   wire    [WB_ADR_WIDTH-1:0]          s_wb_adr_i,
            input   wire    [WB_DAT_WIDTH-1:0]          s_wb_dat_i,
            output  wire    [WB_DAT_WIDTH-1:0]          s_wb_dat_o,
            input   wire                                s_wb_we_i,
            input   wire    [WB_SEL_WIDTH-1:0]          s_wb_sel_i,
            input   wire                                s_wb_stb_i,
            output  wire                                s_wb_ack_o
        );
    
    
    
    // binarizer
    wire    [TUSER_WIDTH-1:0]   axi4s_bin_tuser;
    wire                        axi4s_bin_tlast;
    wire    [0:0]               axi4s_bin_tbinary;
    wire    [DATA_WIDTH-1:0]    axi4s_bin_tdata;
    wire                        axi4s_bin_tvalid;
    wire                        axi4s_bin_tready;
    
    jelly_video_binarizer_core
            #(
                .TUSER_WIDTH        (TUSER_WIDTH),
                .TDATA_WIDTH        (S_TDATA_WIDTH)
            )
        i_video_binarizer_core
            (
                .aresetn            (aresetn),
                .aclk               (aclk),
                
                .param_th           (8'd127),
                .param_inv          (1'b0),
                
                .s_axi4s_tuser      (s_axi4s_tuser),
                .s_axi4s_tlast      (s_axi4s_tlast),
                .s_axi4s_tdata      (s_axi4s_tdata),
                .s_axi4s_tvalid     (s_axi4s_tvalid),
                .s_axi4s_tready     (s_axi4s_tready),
                
                .m_axi4s_tuser      (axi4s_bin_tuser),
                .m_axi4s_tlast      (axi4s_bin_tlast),
                .m_axi4s_tbinary    (axi4s_bin_tbinary),
                .m_axi4s_tdata      (axi4s_bin_tdata),
                .m_axi4s_tvalid     (axi4s_bin_tvalid),
                .m_axi4s_tready     (axi4s_bin_tready)
            );
    
    assign s_wb_ack_o = s_wb_stb_i;
    
    
    
    // DNN
    wire    [TUSER_WIDTH-1:0]           axi4s_dnn_tuser;
    wire                                axi4s_dnn_tlast;
    wire    [M_CLUSTERING_WIDTH-1:0]    axi4s_dnn_tclustering;
    wire                                axi4s_dnn_tvalid;
    wire                                axi4s_dnn_tready;
    
    video_mnist_cnn_core
            #(
                .MAX_X_NUM          (MAX_X_NUM),
                .RAM_TYPE           (RAM_TYPE),
                
                .IMG_Y_NUM          (IMG_Y_NUM),
                .IMG_Y_WIDTH        (IMG_Y_WIDTH),
                
                .S_TDATA_WIDTH      (1),
                .M_TDATA_WIDTH      (M_CLUSTERING_WIDTH),
                .TUSER_WIDTH        (TUSER_WIDTH),
                
                .DEVICE             (DEVICE)
            )
        i_video_mnist_cnn_core
            (
                .aresetn            (aresetn),
                .aclk               (aclk),
                
                .param_blank_num    (8'd3),
                
                .s_axi4s_tuser      (axi4s_bin_tuser),
                .s_axi4s_tlast      (axi4s_bin_tlast),
                .s_axi4s_tdata      (axi4s_bin_tbinary),
                .s_axi4s_tvalid     (axi4s_bin_tvalid),
                .s_axi4s_tready     (axi4s_bin_tready),
                
                .m_axi4s_tuser      (axi4s_dnn_tuser),
                .m_axi4s_tlast      (axi4s_dnn_tlast),
                .m_axi4s_tdata      (axi4s_dnn_tclustering),
                .m_axi4s_tvalid     (axi4s_dnn_tvalid),
                .m_axi4s_tready     (axi4s_dnn_tready)
            );
    
    
    video_dnn_max_count
            #(
                .NUM_CALSS          (NUM_CALSS),
                .CHANNEL_WIDTH      (CHANNEL_WIDTH),
                .TUSER_WIDTH        (TUSER_WIDTH),
                .TNUMBER_WIDTH      (4),
                .TCOUNT_WIDTH       (4)
            )
        i_video_dnn_max_count
            (
                .aresetn            (aresetn),
                .aclk               (aclk),
                
                .s_axi4s_tuser      (axi4s_dnn_tuser),
                .s_axi4s_tlast      (axi4s_dnn_tlast),
                .s_axi4s_tdata      (axi4s_dnn_tclustering),
                .s_axi4s_tvalid     (axi4s_dnn_tvalid),
                .s_axi4s_tready     (axi4s_dnn_tready),
                
                .m_axi4s_tuser      (m_axi4s_tuser),
                .m_axi4s_tlast      (m_axi4s_tlast),
                .m_axi4s_tnumber    (m_axi4s_tnumber),
                .m_axi4s_tcount     (m_axi4s_tcount),
                .m_axi4s_tdata      (m_axi4s_tclustering),
                .m_axi4s_tvalid     (m_axi4s_tvalid),
                .m_axi4s_tready     (m_axi4s_tready)
            );
    
    
endmodule



`default_nettype wire



// end of file
