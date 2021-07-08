// ---------------------------------------------------------------------------
//  MNIST sample
//
//                                 Copyright (C) 2008-2021 by Ryuji Fuchikami
// ---------------------------------------------------------------------------



`timescale 1ns / 1ps
`default_nettype none


module video_mnist_semantic_segmentation
        #(
            parameter   DATA_WIDTH         = 8,
            parameter   RAM_TYPE           = "block",
            
            parameter   MAX_X_NUM          = 1024,
            parameter   IMG_X_WIDTH        = 10,
            parameter   IMG_Y_NUM          = 480,
            parameter   IMG_Y_WIDTH        = 10,
            
            parameter   NUM_CALSS          = 11,
            parameter   CHANNEL_WIDTH      = 1,
            parameter   TUSER_WIDTH        = 1,
            parameter   S_TDATA_WIDTH      = DATA_WIDTH,
            parameter   M_TNUMBER_WIDTH    = 4,
            parameter   M_TCOUNT_WIDTH     = 1,
            parameter   M_CLUSTERING_WIDTH = NUM_CALSS * CHANNEL_WIDTH,
            
            parameter   DEVICE             = "rtl"
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
            input   wire                                m_axi4s_tready
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
    
    
    
    // Convolution layers
    wire    [TUSER_WIDTH-1:0]   axi4s_cnv_tuser;
    wire                        axi4s_cnv_tlast;
    wire    [36-1:0]            axi4s_cnv_tdata;
    wire                        axi4s_cnv_tvalid;
    wire                        axi4s_cnv_tready;
    
    MnistConv
            #(
                .TUSER_WIDTH        (TUSER_WIDTH),
                .IMG_X_WIDTH        (IMG_X_WIDTH),
                .IMG_Y_WIDTH        (IMG_Y_WIDTH),
                .IMG_Y_NUM          (IMG_Y_NUM),
                .MAX_X_NUM          (MAX_X_NUM),
                .BLANK_Y_WIDTH      (8),
                .INIT_Y_NUM         (IMG_Y_NUM),
                .FIFO_PTR_WIDTH     (9),
                .FIFO_RAM_TYPE      ("block"),
                .RAM_TYPE           (RAM_TYPE),
                .DEVICE             (DEVICE)
            )
        i_MnistConv
            (
                .reset              (~aresetn),
                .clk                (aclk),
                
                .param_blank_num    (8'd58),
                
                .s_axi4s_tuser      (axi4s_bin_tuser),
                .s_axi4s_tlast      (axi4s_bin_tlast),
                .s_axi4s_tdata      (axi4s_bin_tbinary),
                .s_axi4s_tvalid     (axi4s_bin_tvalid),
                .s_axi4s_tready     (axi4s_bin_tready),
                
                .m_axi4s_tuser      (axi4s_cnv_tuser),
                .m_axi4s_tlast      (axi4s_cnv_tlast),
                .m_axi4s_tdata      (axi4s_cnv_tdata),
                .m_axi4s_tvalid     (axi4s_cnv_tvalid),
                .m_axi4s_tready     (axi4s_cnv_tready)
            );
    
    
    
    // Classification layers
    wire    [TUSER_WIDTH-1:0]   axi4s_cls_tuser;
    wire                        axi4s_cls_tlast;
    wire    [10-1:0]            axi4s_cls_tdata;
    wire                        axi4s_cls_tvalid;
    wire                        axi4s_cls_tready;
    
    MnistClassification
            #(
                .TUSER_WIDTH        (TUSER_WIDTH),
                .IMG_X_WIDTH        (IMG_X_WIDTH),
                .IMG_Y_WIDTH        (IMG_Y_WIDTH),
                .IMG_Y_NUM          (IMG_Y_NUM),
                .MAX_X_NUM          (MAX_X_NUM),
                .BLANK_Y_WIDTH      (8),
                .INIT_Y_NUM         (IMG_Y_NUM),
                .FIFO_PTR_WIDTH     (9),
                .FIFO_RAM_TYPE      ("block"),
                .RAM_TYPE           (RAM_TYPE),
                .DEVICE             (DEVICE)
            )
        i_MnistClassification
            (
                .reset              (~aresetn),
                .clk                (aclk),
                
                .param_blank_num    (8'd0),
                
                .s_axi4s_tuser      (axi4s_cnv_tuser),
                .s_axi4s_tlast      (axi4s_cnv_tlast),
                .s_axi4s_tdata      (axi4s_cnv_tdata),
                .s_axi4s_tvalid     (axi4s_cnv_tvalid),
                .s_axi4s_tready     (axi4s_cnv_tready),
                
                .m_axi4s_tuser      (axi4s_cls_tuser),
                .m_axi4s_tlast      (axi4s_cls_tlast),
                .m_axi4s_tdata      (axi4s_cls_tdata),
                .m_axi4s_tvalid     (axi4s_cls_tvalid),
                .m_axi4s_tready     (axi4s_cls_tready)
            );
    
    
    // Segmentation layers
    wire    [0:0]               axi4s_seg_tdata;
    
    MnistSegmentation
            #(
                .TUSER_WIDTH        (1),
                .IMG_X_WIDTH        (IMG_X_WIDTH),
                .IMG_Y_WIDTH        (IMG_Y_WIDTH),
                .IMG_Y_NUM          (IMG_Y_NUM),
                .MAX_X_NUM          (MAX_X_NUM),
                .BLANK_Y_WIDTH      (8),
                .INIT_Y_NUM         (IMG_Y_NUM),
                .FIFO_PTR_WIDTH     (9),
                .FIFO_RAM_TYPE      ("block"),
                .RAM_TYPE           (RAM_TYPE),
                .DEVICE             (DEVICE)
            )
        i_MnistSegmentation
            (
                .reset              (~aresetn),
                .clk                (aclk),
                
                .param_blank_num    (8'd0),
                
                .s_axi4s_tuser      (1'b0),
                .s_axi4s_tlast      (axi4s_cnv_tlast),
                .s_axi4s_tdata      (axi4s_cnv_tdata),
                .s_axi4s_tvalid     (axi4s_cnv_tvalid),
                .s_axi4s_tready     (axi4s_cnv_tready),
                
                .m_axi4s_tuser      (),
                .m_axi4s_tlast      (),
                .m_axi4s_tdata      (axi4s_seg_tdata),
                .m_axi4s_tvalid     (),
                .m_axi4s_tready     (axi4s_cls_tready)
            );
    
    
    // 11bit目に数字以外という分類を生成
    wire    [TUSER_WIDTH-1:0]   axi4s_dnn_tuser;
    wire                        axi4s_dnn_tlast;
    wire    [9:0]               axi4s_dnn_tcls;
    wire    [0:0]               axi4s_dnn_tseg;
    wire                        axi4s_dnn_tvalid;
    wire                        axi4s_dnn_tready;
    
    assign axi4s_dnn_tuser   = axi4s_cls_tuser;
    assign axi4s_dnn_tlast   = axi4s_cls_tlast;
    assign axi4s_dnn_tcls    = axi4s_seg_tdata ? axi4s_cls_tdata : 10'd0;
    assign axi4s_dnn_tseg    = ~axi4s_seg_tdata;
    assign axi4s_dnn_tvalid  = axi4s_cls_tvalid;
    assign axi4s_cls_tready  = axi4s_dnn_tready;
    
    
    // 
    video_dnn_max_count
            #(
                .NUM_CALSS          (NUM_CALSS),
                .CHANNEL_WIDTH      (CHANNEL_WIDTH),
                .TUSER_WIDTH        (TUSER_WIDTH),
                .TNUMBER_WIDTH      (4),
                .TCOUNT_WIDTH       (1)
            )
        i_video_dnn_max_count
            (
                .aresetn            (aresetn),
                .aclk               (aclk),
                
                .s_axi4s_tuser      (axi4s_dnn_tuser),
                .s_axi4s_tlast      (axi4s_dnn_tlast),
                .s_axi4s_tdata      ({axi4s_dnn_tseg, axi4s_dnn_tcls}),
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
