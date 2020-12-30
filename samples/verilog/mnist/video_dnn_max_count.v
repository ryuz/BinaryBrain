// ---------------------------------------------------------------------------
//  Jelly  -- the soft-core processor system
//   math
//
//                                 Copyright (C) 2008-2018 by Ryuji Fuchikami
//                                 http://ryuz.my.coocan.jp/
// ---------------------------------------------------------------------------



`timescale 1ns / 1ps
`default_nettype none


module video_dnn_max_count
        #(
            parameter   NUM_CALSS     = 10,
            parameter   CHANNEL_WIDTH = 8,
            
            parameter   TUSER_WIDTH   = 1,
            parameter   TDATA_WIDTH   = CHANNEL_WIDTH*NUM_CALSS,
            parameter   TNUMBER_WIDTH = 4,
            parameter   TCOUNT_WIDTH  = 4
        )
        (
            input   wire                            aresetn,
            input   wire                            aclk,
            
            input   wire    [TUSER_WIDTH-1:0]       s_axi4s_tuser,
            input   wire                            s_axi4s_tlast,
            input   wire    [TDATA_WIDTH-1:0]       s_axi4s_tdata,
            input   wire                            s_axi4s_tvalid,
            output  wire                            s_axi4s_tready,
            
            output  wire    [TUSER_WIDTH-1:0]       m_axi4s_tuser,
            output  wire                            m_axi4s_tlast,
            output  wire    [TNUMBER_WIDTH-1:0]     m_axi4s_tnumber,
            output  wire    [TCOUNT_WIDTH-1:0]      m_axi4s_tcount,
            output  wire    [TDATA_WIDTH-1:0]       m_axi4s_tdata,
            output  wire                            m_axi4s_tvalid,
            input   wire                            m_axi4s_tready
        );
    
    
    wire                                    cke;
    
    
    // counting
    integer                                 i, j;
    integer                                 sum;
    
    reg                                     counting_tlast;
    reg     [TUSER_WIDTH-1:0]               counting_tuser;
    reg     [TDATA_WIDTH-1:0]               counting_tdata;
    reg     [TCOUNT_WIDTH*NUM_CALSS-1:0]    counting_count;
    reg                                     counting_valid;
    always @(posedge aclk) begin
        if( ~aresetn ) begin
            counting_tlast  <= 1'bx;
            counting_tuser  <= {TUSER_WIDTH{1'bx}};
            counting_count  <= {(TCOUNT_WIDTH*NUM_CALSS){1'bx}};
            counting_valid  <= 1'b0;
        end
        else if ( cke ) begin
            counting_tlast <= s_axi4s_tlast;
            counting_tuser <= s_axi4s_tuser;
            counting_tdata <= s_axi4s_tdata;
            for ( i = 0; i < NUM_CALSS; i = i+1 ) begin
                sum = 0;
                for ( j = 0; j < CHANNEL_WIDTH; j = j+1 ) begin
                    sum = sum + s_axi4s_tdata[j*NUM_CALSS + i];
                end
                counting_count[TCOUNT_WIDTH*i +:TCOUNT_WIDTH] <= sum;
            end
            counting_valid  <= s_axi4s_tvalid;
        end
    end
    
    
    // select max
    jelly_minmax
            #(
                .NUM                (NUM_CALSS),
                .COMMON_USER_WIDTH  (1+TUSER_WIDTH+TDATA_WIDTH),
                .USER_WIDTH         (0),
                .DATA_WIDTH         (TCOUNT_WIDTH),
                .DATA_SIGNED        (0),
                .CMP_MIN            (0),    // minかmaxか
                .CMP_EQ             (0)     // 同値のとき data0 と data1 どちらを優先するか
            )
        i_minmax
            (
                .reset              (~aresetn),
                .clk                (aclk),
                .cke                (cke),
                
                .s_common_user      ({counting_tlast, counting_tuser, counting_tdata}),
                .s_user             ({NUM_CALSS{1'b0}}),
                .s_data             (counting_count),
                .s_en               ({NUM_CALSS{1'b1}}),
                .s_valid            (counting_valid),
                
                .m_common_user      ({m_axi4s_tlast, m_axi4s_tuser, m_axi4s_tdata}),
                .m_user             (),
                .m_data             (m_axi4s_tcount),
                .m_index            (m_axi4s_tnumber),
                .m_en               (),
                .m_valid            (m_axi4s_tvalid)
            );
    
    assign cke = !m_axi4s_tvalid | m_axi4s_tready;
    
    assign s_axi4s_tready = cke;
    
endmodule



`default_nettype wire



// end of file
