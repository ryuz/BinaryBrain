// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once


#include <iostream>
#include <iomanip>
#include <vector>
#include <sstream>

#include "bb/Sequential.h"
#include "bb/SparseModel.h"
#include "bb/Convolution2d.h"
#include "bb/MaxPooling.h"
#include "bb/StochasticMaxPooling2x2.h"


namespace bb {


// LUT-Network 基本レイヤーのVerilog 出力
inline void ExportVerilog_LutModel(std::ostream& os, std::string module_name, SparseModel const &lut)
{
    index_t node_size      = lut.GetOutputNodeSize();
    
    // モジュール出力
    os <<
        "\n"
        "\n"
        "module " << module_name << "\n"
        "        #(\n"
        "            parameter DEVICE = \"RTL\"\n"
        "        )\n"
        "        (\n"
        "            input  wire         reset,\n"
        "            input  wire         clk,\n"
        "            input  wire         cke,\n"
        "            \n"
        "            input  wire [" << (lut.GetInputNodeSize() - 1) << ":0]  in_data,\n"
        "            output wire [" << (lut.GetOutputNodeSize() - 1) << ":0]  out_data\n"
        "        );\n"
        "\n";


    for (index_t node = 0; node < node_size; node++) {
        index_t lut_input_size = lut.GetNodeConnectionSize(node);
        int     lut_table_size = lut.GetLutTableSize(node);

        if ( 0 && lut_input_size == 6 ) {
            // LUT 出力(Xilinx)
            os <<
                "    \n"
                "    // LUT : " << node << "\n"
                "    \n"
                "    wire lut_" << node << "_out;\n"
                "    \n"
                "    LUT6\n"
                "            #(\n"
                "                .INIT(" << lut_table_size << "'b";

            for (int bit = lut_table_size - 1; bit >= 0; --bit ) {
                os << (lut.GetLutTable(node, bit) ? "1" : "0");
            }
            os <<
                ")\n";

            os <<
                "            )\n"
                "        i_lut6_" << node << "\n"
                "            (\n"
                "                .O  (lut_" << node << "_out),\n"
                "                .I0 (in_data[" << lut.GetNodeConnectionIndex(node, 0) << "]),\n"
                "                .I1 (in_data[" << lut.GetNodeConnectionIndex(node, 1) << "]),\n"
                "                .I2 (in_data[" << lut.GetNodeConnectionIndex(node, 2) << "]),\n"
                "                .I3 (in_data[" << lut.GetNodeConnectionIndex(node, 3) << "]),\n"
                "                .I4 (in_data[" << lut.GetNodeConnectionIndex(node, 4) << "]),\n"
                "                .I5 (in_data[" << lut.GetNodeConnectionIndex(node, 5) << "])\n";
            os <<
                "            );\n"
                "\n";
        }
        else {
            // LUT 出力
            os <<
                "    \n"
                "    // LUT : " << node << "\n"
                "    \n"
                "    wire lut_" << node << "_out;\n"
                "    \n"
                "    bb_lut\n"
                "            #(\n"
                "                .N(" << lut_input_size << "),\n"
                "                .INIT(" << lut_table_size << "'b";

            for (int bit = lut_table_size - 1; bit >= 0; --bit ) {
                os << (lut.GetLutTable(node, bit) ? "1" : "0");
            }
            os <<
                "),\n"
                "                .DEVICE(DEVICE)\n";

            os <<
                "            )\n"
                "        i_lut_" << node << "\n"
                "            (\n"
                "                .in_data({\n";

            for (index_t bit = lut_input_size - 1; bit >= 1; --bit) {
                os <<
                    "                             in_data[" << lut.GetNodeConnectionIndex(node, bit) << "],\n";
            }
            os <<
                "                             in_data[" << lut.GetNodeConnectionIndex(node, 0) << "]\n"
                "                        }),\n"
                "                .out_data(lut_" << node << "_out)\n"
                "            );\n"
                "    \n";
        }

        os <<
            "    reg   lut_" << node << "_ff;\n"
            "    always @(posedge clk) begin\n"
            "        if ( reset ) begin\n"
            "            lut_" << node << "_ff <= 1'b0;\n"
            "        end\n"
            "        else if ( cke ) begin\n"
            "            lut_" << node << "_ff <= lut_" << node << "_out;\n"
            "        end\n"
            "    end\n"
            "    \n"
            "    assign out_data[" << node << "] = lut_" << node << "_ff;\n"
            "    \n";

        os <<
            "    \n"
            "    \n";
    }

    os <<
        "endmodule\n";
    os << std::endl;
}



// LUT-Network 基本レイヤーの直列接続を出力
inline void ExportVerilog_LutModels(std::ostream& os, std::string module_name, std::vector< std::shared_ptr< SparseModel > > layers)
{
    int layer_size = (int)layers.size();
    BB_ASSERT(layer_size >= 1);

    std::vector<std::string> sub_modle_name;
    auto first_layer = layers[0];
    auto last_layer  = layers[layer_size - 1];

    // サブモジュール名生成
    for (int i = 0; i < layer_size; ++i) {
        std::stringstream ss_sub_name;
        ss_sub_name << module_name << "_sub" << i;
        sub_modle_name.push_back(ss_sub_name.str());
    }
    
    // モジュール出力
    os <<
        "\n"
        "\n"
        "module " << module_name << "\n"
        "        #(\n"
        "            parameter USER_WIDTH = 0,\n"
        "            parameter DEVICE     = \"RTL\",\n"
        "            \n"
        "            parameter USER_BITS  = USER_WIDTH > 0 ? USER_WIDTH : 1\n"
        "        )\n"
        "        (\n"
        "            input  wire                  reset,\n"
        "            input  wire                  clk,\n"
        "            input  wire                  cke,\n"
        "            \n"
        "            input  wire [USER_BITS-1:0]  in_user,\n"
        "            input  wire [" << std::setw(9) << first_layer->GetInputNodeSize() << "-1:0]  in_data,\n"
        "            input  wire                  in_valid,\n"
        "            \n"
        "            output wire [USER_BITS-1:0]  out_user,\n"
        "            output wire [" << std::setw(9) << last_layer->GetOutputNodeSize() << "-1:0]  out_data,\n"
        "            output wire                  out_valid\n"
        "        );\n"
        "\n\n";

    for (int i = 0; i < layer_size; ++i) {
        auto layer = layers[i];

        os
            << "    reg   [USER_BITS-1:0]  layer" << i << "_user;\n"
            << "    wire  [" << std::setw(9) << layer->GetOutputNodeSize() << "-1:0]  layer" << i << "_data;\n"
            << "    reg                    layer" << i << "_valid;\n"
            << "\n"
            << sub_modle_name[i] << "\n"
            << "            #(\n"
            << "                .DEVICE     (DEVICE)\n"
            << "            )\n"
            << "        i_" << sub_modle_name[i] << "\n"
            << "            (\n"
            << "                .reset      (reset),\n"
            << "                .clk        (clk),\n"
            << "                .cke        (cke),\n"
            << "                \n";
        if (i == 0) {
            os << "                .in_data    (in_data),\n";
        }
        else {
            os << "                .in_data    (layer" << (i - 1) << "_data),\n";
        }
        os
            << "                .out_data   (layer" << i << "_data)\n"
            << "             );\n"
            << "    \n"
            << "    always @(posedge clk) begin\n"
            << "        if ( reset ) begin\n"
            << "            layer" << i << "_user  <= {USER_BITS{1'bx}};\n"
            << "            layer" << i << "_valid <= 1'b0;\n"
            << "        end\n"
            << "        else if ( cke ) begin\n";
        if (i == 0) {
            os
                << "            layer" << i << "_user  <= in_user;\n"
                << "            layer" << i << "_valid <= in_valid;\n";
        }
        else {
            os
                << "            layer" << i << "_user  <= layer" << (i - 1) << "_user;\n"
                << "            layer" << i << "_valid <= layer" << (i - 1) << "_valid;\n";
        }
        os
            << "        end\n"
            << "    end\n"
            << "    \n    \n";
    }

    os
        << "    assign out_data  = layer" << (layer_size - 1) << "_data;\n"
        << "    assign out_user  = layer" << (layer_size - 1) << "_user;\n"
        << "    assign out_valid = layer" << (layer_size - 1) << "_valid;\n"
        << "    \n"
        << "endmodule\n"
        << "\n\n";
    

    // サブモジュール出力
    for (int i = 0; i < layer_size; ++i) {
        auto layer = layers[i];
        ExportVerilog_LutModel(os, sub_modle_name[i], *layer);
    }
}


// LUT-Network 基本レイヤーの直列接続を出力
inline void ExportVerilog_LutModels(std::ostream& os, std::string module_name, std::shared_ptr<bb::Sequential> net)
{
    std::vector< std::shared_ptr< SparseModel > > layers;

    // LutModel だけを取り出し
    for (int i = 0; i < net->GetSize(); ++i) {
        auto layer = std::dynamic_pointer_cast< SparseModel >(net->Get(i));
        if ( layer != nullptr ) {
            layers.push_back(layer);
        }
    }

    ExportVerilog_LutModels(os, module_name, layers);
}


inline void ExportVerilog_LutModels(std::ostream& os, std::string module_name, std::vector< std::shared_ptr< Model > > layers)
{
    std::vector< std::shared_ptr< SparseModel > > sparse_layers;
    for (auto model : layers) {
        auto sparse_model = std::dynamic_pointer_cast< SparseModel >(model);
        if (sparse_model) {
            sparse_layers.push_back(sparse_model);
        }
    }
    ExportVerilog_LutModels(os, module_name, sparse_layers);
}




// Convolutionモジュールの出力
inline void ExportVerilog_LutConvolutionModule(std::ostream& os, std::string module_name, std::string mlp_name, int in_c, int out_c, int n, int m)
{
    os << "\n\n\n";
    os << "module " << module_name << "\n";

    os << R"(        #(
            parameter   USER_WIDTH = 0,
            parameter   MAX_X_NUM  = 1024,
            parameter   RAM_TYPE   = "block",
            parameter   DEVICE     = "rtl",
)";
    os << "            parameter   S_C  = " << in_c << ",\n";
    os << "            parameter   M_C  = " << out_c << ",\n";
    os << "            parameter   N  = " << n << ",\n";
    os << "            parameter   M  = " << m << ",\n";
            
    os << R"(            parameter   USER_BITS  = USER_WIDTH > 0 ? USER_WIDTH : 1
        )
        (
            input   wire                            reset,
            input   wire                            clk,
            input   wire                            cke,
            
            input   wire                            s_img_line_first,
            input   wire                            s_img_line_last,
            input   wire                            s_img_pixel_first,
            input   wire                            s_img_pixel_last,
            input   wire                            s_img_de,
            input   wire    [USER_BITS-1:0]         s_img_user,
            input   wire    [S_C-1:0]               s_img_data,
            input   wire                            s_img_valid,
            
            output  wire                            m_img_line_first,
            output  wire                            m_img_line_last,
            output  wire                            m_img_pixel_first,
            output  wire                            m_img_pixel_last,
            output  wire                            m_img_de,
            output  wire    [USER_BITS-1:0]         m_img_user,
            output  wire    [M_C-1:0]               m_img_data,
            output  wire                            m_img_valid
        );
)";

    
    os << R"(
    localparam  NC = (N-1) / 2;
    localparam  MC = (M-1) / 2;
    
    
    wire                            img_blk_line_first;
    wire                            img_blk_line_last;
    wire                            img_blk_pixel_first;
    wire                            img_blk_pixel_last;
    wire                            img_blk_de;
    wire    [USER_BITS-1:0]         img_blk_user;
    wire    [N*M*S_C-1:0]           img_blk_data;
    wire                            img_blk_valid;
    
    jelly_img_blk_buffer
            #(
                .USER_WIDTH         (USER_WIDTH),
                .DATA_WIDTH         (S_C),
                .LINE_NUM           (N),
                .PIXEL_NUM          (M),
                .LINE_CENTER        (NC),
                .PIXEL_CENTER       (MC),
                .MAX_X_NUM          (MAX_X_NUM),
                .RAM_TYPE           (RAM_TYPE),
                .BORDER_MODE        ("CONSTANT"),
                .BORDER_VALUE       ({(N*M){1'b0}})
            )
        i_img_blk_buffer
            (
                .reset              (reset),
                .clk                (clk),
                .cke                (cke),
                
                .s_img_line_first   (s_img_line_first),
                .s_img_line_last    (s_img_line_last),
                .s_img_pixel_first  (s_img_pixel_first),
                .s_img_pixel_last   (s_img_pixel_last),
                .s_img_de           (s_img_de),
                .s_img_user         (s_img_user),
                .s_img_data         (s_img_data),
                .s_img_valid        (s_img_valid),
                
                .m_img_line_first   (img_blk_line_first),
                .m_img_line_last    (img_blk_line_last),
                .m_img_pixel_first  (img_blk_pixel_first),
                .m_img_pixel_last   (img_blk_pixel_last),
                .m_img_de           (img_blk_de),
                .m_img_user         (img_blk_user),
                .m_img_data         (img_blk_data),
                .m_img_valid        (img_blk_valid)
            );
    
    genvar                          i, j, k;
    wire    [N*M*S_C-1:0]           img_blk_data_shuffle;
    generate
    for ( i = 0; i < S_C; i = i+1 ) begin : loop_i
        for ( j = 0; j < N; j = j+1 ) begin : loop_j
            for ( k = 0; k < M; k = k+1 ) begin : loop_j
                assign img_blk_data_shuffle[i*(N*M) + j*M + k] = img_blk_data[(j*M + k)*S_C + i];
            end
        end
    end
    endgenerate

)";

    os << "\n\n";
    os << "    " << mlp_name << "\n";
    os << R"(
            #(
                .USER_WIDTH (USER_BITS + 5),
                .DEVICE     (DEVICE)
            )
        i_mlp
            (
                .reset      (reset),
                .clk        (clk),
                .cke        (cke),
            
                .in_user    ({
                                img_blk_user,
                                img_blk_line_first,
                                img_blk_line_last,
                                img_blk_pixel_first,
                                img_blk_pixel_last,
                                img_blk_de
                            }),
                .in_data    (img_blk_data_shuffle),
                .in_valid   (img_blk_valid),
            
                .out_user   ({
                                m_img_user,
                                m_img_line_first,
                                m_img_line_last,
                                m_img_pixel_first,
                                m_img_pixel_last,
                                m_img_de
                            }),
                .out_data   (m_img_data),
                .out_valid  (m_img_valid)
            );


endmodule
)";


}


inline void ExportVerilog_LutConvolutionLayer(std::ostream& os, std::string module_name, std::shared_ptr< Filter2d > conv)
{
    auto sub_layer = conv->GetSubLayer();
    if ( !sub_layer ) {
        std::cout << "error : Convolution2d don't have sub layer" << std::endl;
        BB_ASSERT(0);
        return;
    }

    auto seq_model = std::dynamic_pointer_cast<Sequential>(sub_layer);
    if ( !seq_model ) {
        seq_model = Sequential::Create();
        seq_model->Add(sub_layer);
    }

    std::string sub_name = module_name + "_sub";

    auto in_shape  = conv->GetInputShape();
    auto out_shape = conv->GetOutputShape();
    BB_ASSERT(in_shape.size() == 3);
    BB_ASSERT(out_shape.size() == 3);

    int in_c  = (int)in_shape[0];
    int out_c = (int)out_shape[0];
    int n = (int)conv->GetFilterHeight();
    int m = (int)conv->GetFilterWidth();

    ExportVerilog_LutConvolutionModule(os, module_name, sub_name, in_c, out_c, n, m);
    ExportVerilog_LutModels(os, sub_name, seq_model);
}


inline void ExportVerilog_LutCnnLayersAxi4s(std::ostream& os, std::string module_name, std::vector< std::shared_ptr< Filter2d > > layers)
{
    int  layer_size = (int)layers.size();
    auto fisrt_layer = layers[0];
    auto last_layer = layers[layer_size - 1];

    auto in_shape  = fisrt_layer->GetInputShape();
    auto out_shape = last_layer->GetOutputShape();
    BB_ASSERT(in_shape.size() == 3);
    BB_ASSERT(out_shape.size() == 3);
    int in_c  = (int)in_shape[0];
    int out_c = (int)out_shape[0];


    os << "module " << module_name << "\n"; 
    os << R"(
        #(
            parameter   TUSER_WIDTH    = 1,
            parameter   IMG_X_WIDTH    = 10,
            parameter   IMG_Y_WIDTH    = 9,
            parameter   IMG_Y_NUM      = 480,
            parameter   MAX_X_NUM      = 1024,
            parameter   BLANK_Y_WIDTH  = 8,
            parameter   INIT_Y_NUM     = IMG_Y_NUM,
            parameter   FIFO_PTR_WIDTH = 9,
            parameter   FIFO_RAM_TYPE  = "block",
            parameter   RAM_TYPE       = "block",
            parameter   IMG_CKE_BUFG   = 0,
            parameter   DEVICE         = "rtl",
)";

    os << "            parameter   S_TDATA_WIDTH  = " << in_c << ",\n";
    os << "            parameter   M_TDATA_WIDTH  = " << out_c;

    os << R"(
        )
        (
            input   wire                                reset,
            input   wire                                clk,
            
            input   wire    [BLANK_Y_WIDTH-1:0]         param_blank_num,
            
            input   wire    [TUSER_WIDTH-1:0]           s_axi4s_tuser,
            input   wire                                s_axi4s_tlast,
            input   wire    [S_TDATA_WIDTH-1:0]         s_axi4s_tdata,
            input   wire                                s_axi4s_tvalid,
            output  wire                                s_axi4s_tready,
            
            output  wire    [TUSER_WIDTH-1:0]           m_axi4s_tuser,
            output  wire                                m_axi4s_tlast,
            output  wire    [M_TDATA_WIDTH-1:0]         m_axi4s_tdata,
            output  wire                                m_axi4s_tvalid,
            input   wire                                m_axi4s_tready
        );
)";

    os << R"(

    localparam  USER_WIDTH = TUSER_WIDTH > 1 ? TUSER_WIDTH - 1 : 1;

    wire                                cke;
    
    wire                                src_img_line_first;
    wire                                src_img_line_last;
    wire                                src_img_pixel_first;
    wire                                src_img_pixel_last;
    wire                                src_img_de;
    wire    [USER_WIDTH-1:0]            src_img_user;
    wire    [S_TDATA_WIDTH-1:0]         src_img_data;
    wire                                src_img_valid;
    
    wire                                sink_img_line_first;
    wire                                sink_img_line_last;
    wire                                sink_img_pixel_first;
    wire                                sink_img_pixel_last;
    wire                                sink_img_de;
    wire    [USER_WIDTH-1:0]            sink_img_user;
    wire    [M_TDATA_WIDTH-1:0]         sink_img_data;
    wire                                sink_img_valid;
    
    jelly_axi4s_img
            #(
                .TUSER_WIDTH            (TUSER_WIDTH),
                .S_TDATA_WIDTH          (S_TDATA_WIDTH),
                .M_TDATA_WIDTH          (M_TDATA_WIDTH),
                .IMG_X_WIDTH            (IMG_X_WIDTH),
                .IMG_Y_WIDTH            (IMG_Y_WIDTH),
                .IMG_Y_NUM              (IMG_Y_NUM),
                .USE_DE                 (1),
                .USE_VALID              (1),
                .BLANK_Y_WIDTH          (BLANK_Y_WIDTH),
                .INIT_Y_NUM             (INIT_Y_NUM),
                .FIFO_PTR_WIDTH         (FIFO_PTR_WIDTH),
                .FIFO_RAM_TYPE          (FIFO_RAM_TYPE),
                .IMG_CKE_BUFG           (IMG_CKE_BUFG)
            )
        i_axi4s_img
            (
                .reset                  (reset),
                .clk                    (clk),
                
                .param_blank_num        (param_blank_num),
                
                .s_axi4s_tuser          (s_axi4s_tuser),
                .s_axi4s_tlast          (s_axi4s_tlast),
                .s_axi4s_tdata          (s_axi4s_tdata),
                .s_axi4s_tvalid         (s_axi4s_tvalid),
                .s_axi4s_tready         (s_axi4s_tready),
                
                .m_axi4s_tuser          (m_axi4s_tuser),
                .m_axi4s_tlast          (m_axi4s_tlast),
                .m_axi4s_tdata          (m_axi4s_tdata),
                .m_axi4s_tvalid         (m_axi4s_tvalid),
                .m_axi4s_tready         (m_axi4s_tready),
                
                
                .img_cke                (cke),
                
                .src_img_line_first     (src_img_line_first),
                .src_img_line_last      (src_img_line_last),
                .src_img_pixel_first    (src_img_pixel_first),
                .src_img_pixel_last     (src_img_pixel_last),
                .src_img_de             (src_img_de),
                .src_img_user           (src_img_user),
                .src_img_data           (src_img_data),
                .src_img_valid          (src_img_valid),
                
                .sink_img_line_first    (sink_img_line_first),
                .sink_img_line_last     (sink_img_line_last),
                .sink_img_pixel_first   (sink_img_pixel_first),
                .sink_img_pixel_last    (sink_img_pixel_last),
                .sink_img_de            (sink_img_de),
                .sink_img_user          (sink_img_user),
                .sink_img_data          (sink_img_data),
                .sink_img_valid         (sink_img_valid)
            );
    
    
)";

    os << "    localparam DATA0_WIDTH = " << in_c << ";\n";
    for ( int i = 0; i < layer_size; ++i ) {
        os << "    localparam DATA" << i+1 << "_WIDTH = " << layers[i]->GetOutputChannels() << ";\n";
    }
    os << "    \n";
    
    for ( int i = 0; i < layer_size+1; ++i ) {
        os << "    \n";
        os << "    wire                           img" << i << "_line_first;\n";
        os << "    wire                           img" << i << "_line_last;\n";
        os << "    wire                           img" << i << "_pixel_first;\n";
        os << "    wire                           img" << i << "_pixel_last;\n";
        os << "    wire                           img" << i << "_de;\n";
        os << "    wire   [USER_WIDTH-1:0]        img" << i << "_user;\n";
        os << "    wire   [DATA" << i << "_WIDTH-1:0]       img" << i << "_data;\n";
        os << "    wire                           img" << i << "_valid;\n";
    }


    for ( int i = 0; i < layer_size; ++i ) {
        os << "\n\n";

        auto layer       = layers[i];
        auto layer_class = layer->GetModelName();
//        auto cnv   = std::dynamic_pointer_cast<Convolution2d<FT, BT> >(layer);
//        auto pol   = std::dynamic_pointer_cast<MaxPooling<FT, BT> >(layer);
//        auto pol_s = std::dynamic_pointer_cast< StochasticMaxPooling2x2<> >(layer);
        if ( layer_class == "Convolution2d" ) {
            os << "    " << module_name << "_l" << i << "\n";
            os << "            #(\n";
            os << "                .USER_WIDTH              (USER_WIDTH),\n";
            os << "                .MAX_X_NUM               (MAX_X_NUM),\n";
            os << "                .RAM_TYPE                (RAM_TYPE),\n";
            os << "                .DEVICE                  (DEVICE)\n";
            os << "            )\n";
            os << "        i_" << module_name << "_l" << i << "\n";
        }
        else if ( layer_class == "MaxPooling"
                    || layer_class == "StochasticMaxPooling"
                    || layer_class == "StochasticMaxPooling2x2" ) {
            os << "    " << "jelly_img_dnn_maxpol" << "\n";
            os << "            #(\n";
            os << "                .C                       (" << layer->GetOutputChannels() << "),\n";
            os << "                .N                       (" << layer->GetFilterWidth() << "),\n";
            os << "                .M                       (" << layer->GetFilterHeight() << "),\n";
            os << "                .USER_WIDTH              (USER_WIDTH),\n";
            os << "                .MAX_X_NUM               (MAX_X_NUM),\n";
            os << "                .RAM_TYPE                (RAM_TYPE)\n";
            os << "            )\n";
            os << "        i_" << "i_img_dnn_maxpol" << "_l" << i << "\n";
        }
        else {
            std::cout << "error : Unknown model" << layer_class << std::endl;
            BB_ASSERT(0);
            return;
        }

        os << "            (\n";
        os << "                .reset                   (reset),\n";
        os << "                .clk                     (clk),\n";
        os << "                .cke                     (cke),\n";
        os << "                \n";
        os << "                .s_img_line_first        (img" << i << "_line_first),\n";
        os << "                .s_img_line_last         (img" << i << "_line_last),\n";
        os << "                .s_img_pixel_first       (img" << i << "_pixel_first),\n";
        os << "                .s_img_pixel_last        (img" << i << "_pixel_last),\n";
        os << "                .s_img_de                (img" << i << "_de),\n";
        os << "                .s_img_user              (img" << i << "_user),\n";
        os << "                .s_img_data              (img" << i << "_data),\n";
        os << "                .s_img_valid             (img" << i << "_valid),\n";
        os << "                \n";
        os << "                .m_img_line_first        (img" << i+1 << "_line_first),\n";
        os << "                .m_img_line_last         (img" << i+1 << "_line_last),\n";
        os << "                .m_img_pixel_first       (img" << i+1 << "_pixel_first),\n";
        os << "                .m_img_pixel_last        (img" << i+1 << "_pixel_last),\n";
        os << "                .m_img_de                (img" << i+1 << "_de),\n";
        os << "                .m_img_user              (img" << i+1 << "_user),\n";
        os << "                .m_img_data              (img" << i+1 << "_data),\n";
        os << "                .m_img_valid             (img" << i+1 << "_valid)\n";
        os << "            );\n";
    }

    os << "    \n";
    os << "    \n";
    os << "    assign img" << 0 << "_line_first  = src_img_line_first;\n";
    os << "    assign img" << 0 << "_line_last   = src_img_line_last;\n";
    os << "    assign img" << 0 << "_pixel_first = src_img_pixel_first;\n";
    os << "    assign img" << 0 << "_pixel_last  = src_img_pixel_last;\n";
    os << "    assign img" << 0 << "_de          = src_img_de;\n";
    os << "    assign img" << 0 << "_user        = src_img_user;\n";
    os << "    assign img" << 0 << "_data        = src_img_data;\n";
    os << "    assign img" << 0 << "_valid       = src_img_valid;\n";
    os << "    \n";
    os << "    assign sink_img_line_first  = img" << layer_size << "_line_first;\n";
    os << "    assign sink_img_line_last   = img" << layer_size << "_line_last;\n";
    os << "    assign sink_img_pixel_first = img" << layer_size << "_pixel_first;\n";
    os << "    assign sink_img_pixel_last  = img" << layer_size << "_pixel_last;\n";
    os << "    assign sink_img_de          = img" << layer_size << "_de;\n";
    os << "    assign sink_img_user        = img" << layer_size << "_user;\n";
    os << "    assign sink_img_data        = img" << layer_size << "_data;\n";
    os << "    assign sink_img_valid       = img" << layer_size << "_valid;\n";

    os << "    \n";
    os << "    \n";
    os << "endmodule\n\n";

    for ( int i = 0; i < layer_size; ++i ) {
        auto layer = layers[i];
        if ( layer->GetModelName() == "Convolution2d" ) {
            std::stringstream ss;
            ss << module_name << "_l" << i;
            ExportVerilog_LutConvolutionLayer(os, ss.str(), layer);
        }
    }
}


inline void ExportVerilog_LutCnnLayersAxi4s(std::ostream& os, std::string module_name, std::vector< std::shared_ptr< Model > > layers)
{
    std::vector< std::shared_ptr< Filter2d > > filters;
    for (auto model : layers) {
        auto filter = std::dynamic_pointer_cast<Filter2d>(model);
        if (filter) {
            filters.push_back(filter);
        }
    }

    if (filters.size() > 0) {
        ExportVerilog_LutCnnLayersAxi4s(os, module_name, filters);
    }
}


}