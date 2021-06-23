// ---------------------------------------------------------------------------
//  MNIST sample
//
//                                 Copyright (C) 2008-2021 by Ryuji Fuchikami
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
    
    reg     reset = 1'b1;
    initial #(RATE*100) reset = 1'b0;
    
    reg     clk = 1'b1;
    always #(RATE/2.0)  clk = ~clk;
    
    wire    cke = 1'b1;
    
    
    
    localparam  FILE_NAME    = "mnist_test.txt";
    localparam  DATA_SIZE    = 10000;
    localparam  USER_WIDTH   = 8;
    localparam  INPUT_WIDTH  = 28*28;
    localparam  CLASS_NUM    = 10;
    localparam  CHANNEL_NUM  = 1;       // チャネル方向(空間的に)多重
    localparam  OUTPUT_WIDTH = CLASS_NUM * CHANNEL_NUM;
    
    reg     [USER_WIDTH+INPUT_WIDTH-1:0]    mem     [0:DATA_SIZE-1];
    initial begin
        $readmemb(FILE_NAME, mem);
    end
    
    
    integer                                 index = 0;
    wire                                    in_last = (index == DATA_SIZE-1);
    wire        [USER_WIDTH-1:0]            in_user;
    wire        [INPUT_WIDTH-1:0]           in_data;
    reg                                     in_valid = 0;
    
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
    
    
    wire                                out_last;
    wire        [USER_WIDTH-1:0]        out_user;
    wire        [OUTPUT_WIDTH-1:0]      out_data;
    wire                                out_valid;
    
    MnistLutSimple
            #(
                .USER_WIDTH     (1+USER_WIDTH)
            )
        i_MnistLutSimple
            (
                .reset          (reset),
                .clk            (clk),
                .cke            (cke),
                
                .in_user        ({in_last, in_user}),
                .in_data        (in_data),
                .in_valid       (in_valid),
                
                .out_user       ({out_last, out_user}),
                .out_data       (out_data),
                .out_valid      (out_valid)
            );
    
    
    // sum
    integer                         i, j;
    integer                         tmp;
    reg                             sum_last;
    reg     [CLASS_NUM*32-1:0]      sum_data;
    reg     [USER_WIDTH-1:0]        sum_user;
    reg                             sum_valid;
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
    
    integer     max_index;
    integer     max_value;
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
    integer     out_data_counter = 0;
    integer     out_ok_counter   = 0;
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
