// ---------------------------------------------------------------------------
//  MNIST sample
//
//                                 Copyright (C) 2008-2021 by Ryuji Fuchikami
// ---------------------------------------------------------------------------


`timescale 1ns / 1ps
`default_nettype none


module tb_mnist_lut_semantic_segmentation(
            input   logic   reset,
            input   logic   clk
        );
    
    /*
    localparam RATE = 1000.0/300.0;
    
    initial begin
        $dumpfile("tb_mnist_lut_semantic_segmentation.vcd");
        $dumpvars(3, tb_mnist_lut_semantic_segmentation);

    #20000000
        $finish;
    end
    
    reg     reset = 1'b1;
    initial #(RATE*100) reset = 1'b0;
    
    reg     clk = 1'b1;
    always #(RATE/2.0)  clk = ~clk;
    */

    wire    cke = 1'b1;
    
    
    localparam PPM_FILE  = "mnist_test_640x480.ppm";
    localparam IMG_X_NUM = 640;
    localparam IMG_Y_NUM = 480;
    
//    localparam PPM_FILE  = "mnist_test_160x120.ppm";
//    localparam IMG_X_NUM = 160;
//    localparam IMG_Y_NUM = 120;
    
    localparam  DATA_WIDTH         = 8;
    
    localparam  IMG_Y_WIDTH        = 12;
    
    localparam  TUSER_WIDTH        = 1;
    localparam  S_TDATA_WIDTH      = 8;
    
    localparam  M_TNUMBER_WIDTH    = 4;
    localparam  M_TCOUNT_WIDTH     = 1;
    localparam  M_CLUSTERING_WIDTH = 11;
    
    localparam  WB_ADR_WIDTH       = 8;
    localparam  WB_DAT_WIDTH       = 32;
    localparam  WB_SEL_WIDTH       = (WB_DAT_WIDTH / 8);
    localparam  INIT_PARAM_TH      = 127;
    localparam  INIT_PARAM_INV     = 1'b0;
    
    
    
    
    // ----------------------------------
    //  dummy video
    // ----------------------------------
    
    wire    [0:0]   s_axi4s_tuser;
    wire            s_axi4s_tlast;
    wire    [31:0]  s_axi4s_tdata;
    wire            s_axi4s_tvalid;
    wire            s_axi4s_tready;
    
    jelly_axi4s_master_model
            #(
                .AXI4S_DATA_WIDTH   (24),
                .X_NUM              (IMG_X_NUM),
                .Y_NUM              (IMG_Y_NUM),
                .PPM_FILE           (PPM_FILE),
                .BUSY_RATE          (0), // (5),
                .RANDOM_SEED        (776),
                .INTERVAL           (1000)
            )
        i_axi4s_master_model
            (
                .aresetn            (~reset),
                .aclk               (clk),
                
                .m_axi4s_tuser      (s_axi4s_tuser),
                .m_axi4s_tlast      (s_axi4s_tlast),
                .m_axi4s_tdata      (s_axi4s_tdata[23:0]),
                .m_axi4s_tvalid     (s_axi4s_tvalid),
                .m_axi4s_tready     (s_axi4s_tready)
            );
    
    assign s_axi4s_tdata[31:24] = 0;
    
    
    
    // ----------------------------------
    //  image dump
    // ----------------------------------
    
    localparam FRAME_NUM = 1;
    
    integer     fp_img0;
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
    
    wire    [TUSER_WIDTH-1:0]           m_axi4s_tuser;
    wire                                m_axi4s_tlast;
    wire    [M_TNUMBER_WIDTH-1:0]       m_axi4s_tnumber;
    wire    [M_TCOUNT_WIDTH-1:0]        m_axi4s_tcount;
    wire    [M_CLUSTERING_WIDTH-1:0]    m_axi4s_tclustering;
    wire                                m_axi4s_tvalid;
    wire                                m_axi4s_tready;
    
    video_mnist_semantic_segmentation
            #(
                .DATA_WIDTH             (DATA_WIDTH),
                .IMG_Y_NUM              (IMG_Y_NUM),
                .IMG_Y_WIDTH            (IMG_Y_WIDTH),
                .TUSER_WIDTH            (TUSER_WIDTH),
                .S_TDATA_WIDTH          (S_TDATA_WIDTH)
            )
        i_video_mnist_semantic_segmentation
            (
                .aresetn                (~reset),
                .aclk                   (clk),
                
                .s_axi4s_tuser          (s_axi4s_tuser),
                .s_axi4s_tlast          (s_axi4s_tlast),
                .s_axi4s_tdata          (s_axi4s_tdata[7:0]),
                .s_axi4s_tvalid         (s_axi4s_tvalid),
                .s_axi4s_tready         (s_axi4s_tready),
                
                .m_axi4s_tuser          (m_axi4s_tuser),
                .m_axi4s_tlast          (m_axi4s_tlast),
                .m_axi4s_tclustering    (m_axi4s_tclustering),
                .m_axi4s_tnumber        (m_axi4s_tnumber),
                .m_axi4s_tcount         (m_axi4s_tcount),
                .m_axi4s_tvalid         (m_axi4s_tvalid),
                .m_axi4s_tready         (m_axi4s_tready)
            );
    
    
    wire    [0:0]               axi4s_color_tuser;
    wire                        axi4s_color_tlast;
    wire    [31:0]              axi4s_color_tdata;
    wire                        axi4s_color_tvalid;
    wire                        axi4s_color_tready;
    
    video_mnist_color
            #(
                .DATA_WIDTH         (DATA_WIDTH),
                .TUSER_WIDTH        (1),
                .TNUMBER_WIDTH      (4),
                .TCOUNT_WIDTH       (1),
                .INIT_PARAM_MODE    (2'b10),
                .INIT_PARAM_TH      (1)
            )
        i_video_mnist_color
            (
                .aresetn            (~reset),
                .aclk               (clk),
                
                .s_axi4s_tuser      (m_axi4s_tuser),
                .s_axi4s_tlast      (m_axi4s_tlast),
                .s_axi4s_tnumber    (m_axi4s_tnumber),
                .s_axi4s_tcount     (m_axi4s_tcount),
                .s_axi4s_tdata      (32'h00202020),     // (m_axi4s_tdata),
                .s_axi4s_tbinary    (1'b0),             // (m_axi4s_tbinary),
                .s_axi4s_tvalid     (m_axi4s_tvalid),
                .s_axi4s_tready     (m_axi4s_tready),
                
                .m_axi4s_tuser      (axi4s_color_tuser),
                .m_axi4s_tlast      (axi4s_color_tlast),
                .m_axi4s_tdata      (axi4s_color_tdata),
                .m_axi4s_tvalid     (axi4s_color_tvalid),
                .m_axi4s_tready     (axi4s_color_tready),
                
                .s_wb_rst_i         (reset),
                .s_wb_clk_i         (clk),
                .s_wb_adr_i         (8'd0),
                .s_wb_dat_i         (32'd0),
                .s_wb_dat_o         (),
                .s_wb_we_i          (1'b0),
                .s_wb_sel_i         (4'd0),
                .s_wb_stb_i         (1'b0),
                .s_wb_ack_o         ()
        );
    
    
    
    // 出力結果を保存
    jelly_axi4s_slave_model
            #(
                .COMPONENT_NUM      (3),
                .DATA_WIDTH         (DATA_WIDTH),
                .INIT_FRAME_NUM     (0),
                .FRAME_WIDTH        (32),
                .X_WIDTH            (32),
                .Y_WIDTH            (32),
                .FILE_NAME          ("col_%1d.ppm"),
                .MAX_PATH           (64),
                .BUSY_RATE          (0),
                .RANDOM_SEED        (1234)
            )
        jelly_axi4s_slave_model_col
            (
                .aresetn            (~reset),
                .aclk               (clk),
                .aclken             (1'b1),
                
                .param_width        (IMG_X_NUM),
                .param_height       (IMG_Y_NUM),
                
                .s_axi4s_tuser      (axi4s_color_tuser),
                .s_axi4s_tlast      (axi4s_color_tlast),
                .s_axi4s_tdata      ({axi4s_color_tdata[7:0], axi4s_color_tdata[15:8], axi4s_color_tdata[23:16]}),
                .s_axi4s_tvalid     (axi4s_color_tvalid),
                .s_axi4s_tready     (axi4s_color_tready)
            );
    
    
    // 出力フレームカウント
    integer output_frame = 0;
    /*
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
    */
    
    
endmodule


`default_nettype wire


// end of file
