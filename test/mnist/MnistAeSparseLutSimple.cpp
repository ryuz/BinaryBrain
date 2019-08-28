// --------------------------------------------------------------------------
//  BinaryBrain  -- binary network evaluation platform
//   MNIST sample
//
//                                Copyright (C) 2018-2019 by Ryuji Fuchikami
// --------------------------------------------------------------------------


#include <iostream>

#include "bb/Sequential.h"
#include "bb/SparseLutN.h"
#include "bb/SparseLutDiscreteN.h"
#include "bb/BinaryLutN.h"
#include "bb/Reduce.h"
#include "bb/BinaryModulation.h"
#include "bb/OptimizerAdam.h"
#include "bb/LossMeanSquaredError.h"
#include "bb/MetricsMeanSquaredError.h"
#include "bb/Runner.h"
#include "bb/LoadMnist.h"
#include "bb/ExportVerilog.h"



static void WritePgm(std::string fname, bb::FrameBuffer buf, int frame)
{
    std::ofstream ofs(fname);
    ofs << "P2\n";
    ofs << "28 28 \n";
    ofs << "255\n";
    for ( int i = 0; i < 28*28; ++i ) {
        ofs << (int)(buf.GetFP32(frame, i) * 255.0f) << "\n";
    }
}


// AutoEncoder
void MnistAeSparseLutSimple(int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read)
{
    std::string net_name = "MnistAeSparseLutSimple";

  // load MNIST data
#ifdef _DEBUG
    auto td = bb::LoadMnist<>::Load(64, 32);
    std::cout << "!!! debug mode !!!" << std::endl;
#else
    auto td = bb::LoadMnist<>::Load();
#endif

    // 入力と出力を同じに
    td.t_shape = td.x_shape;
    td.t_train = td.x_train;
    td.t_test  = td.x_test;

    auto enc_sl0 = bb::SparseLutN<6, bb::Bit>::Create(6912);
    auto enc_sl1 = bb::SparseLutN<6, bb::Bit>::Create(1152);
    auto enc_sl2 = bb::SparseLutN<6, bb::Bit>::Create(192);
    auto enc_sl3 = bb::SparseLutN<6, bb::Bit>::Create(32);
//    auto enc_sl3 = bb::StochasticLutN<6, bb::Bit>::Create(32);
//    auto enc_sl3b = bb::Binarize<bb::Bit>::Create(0.5f, 0.0f, 1.0f);

    auto dec_sl0 = bb::SparseLutN<6, bb::Bit>::Create(28*28*6*6);
    auto dec_sl1 = bb::SparseLutN<6, bb::Bit>::Create(28*28*6);
    auto dec_sl2 = bb::SparseLutN<6, bb::Bit>::Create(28*28, false);
//    auto dec_sl2 = bb::StochasticLutN<6, bb::Bit>::Create(28*28);
//    auto dec_sl2b = bb::Binarize<bb::Bit>::Create(0.5f, 0.0f, 1.0f);

    {
        std::cout << "\n<Training>" << std::endl;

        // main network
        auto main_net = bb::Sequential::Create();
        main_net->Add(enc_sl0);
        main_net->Add(enc_sl1);
        main_net->Add(enc_sl2);
        main_net->Add(enc_sl3);
//        main_net->Add(enc_sl3b);
        main_net->Add(dec_sl0);
        main_net->Add(dec_sl1);
        main_net->Add(dec_sl2);
//        main_net->Add(dec_sl2b);

        // modulation wrapper
        auto net = bb::Sequential::Create();
        net->Add(bb::BinaryModulation<bb::Bit>::Create(main_net, train_modulation_size, test_modulation_size));

        // set input shape
        net->SetInputShape(td.x_shape);

        // set binary mode
        if ( binary_mode ) {
            net->SendCommand("binary true");
        }
        else {
            net->SendCommand("binary false");
        }

        // print model information
        net->PrintInfo();

        std::cout << "-----------------------------------" << std::endl;
        std::cout << "epoch_size            : " << epoch_size            << std::endl;
        std::cout << "mini_batch_size       : " << mini_batch_size       << std::endl;
        if ( binary_mode ) {
        std::cout << "train_modulation_size : " << train_modulation_size << std::endl;
        std::cout << "test_modulation_size  : " << test_modulation_size  << std::endl;
        }
        std::cout << "binary_mode           : " << binary_mode           << std::endl;
        std::cout << "file_read             : " << file_read             << std::endl;
        std::cout << "-----------------------------------" << std::endl;

        // fitting
        bb::Runner<float>::create_t runner_create;
        runner_create.name               = net_name;
        runner_create.net                = net;
        runner_create.lossFunc           = bb::LossMeanSquaredError<float>::Create();
        runner_create.metricsFunc        = bb::MetricsMeanSquaredError<float>::Create();
        runner_create.optimizer          = bb::OptimizerAdam<float>::Create();
        runner_create.file_read          = file_read;       // 前の計算結果があれば読み込んで再開するか
        runner_create.file_write         = true;            // 計算結果をファイルに保存するか
        runner_create.print_progress     = true;            // 途中結果を表示
        runner_create.initial_evaluation = false; // file_read;       // ファイルを読んだ場合は最初に評価しておく
        auto runner = bb::Runner<float>::Create(runner_create);
        runner->Fitting(td, epoch_size, mini_batch_size);

        
        bb::FrameBuffer x_buf(32, {28, 28, 1}, BB_TYPE_FP32);
        x_buf.SetVector(td.x_test, 0);
        auto y_buf = net->Forward(x_buf, false);

        WritePgm("out_0x.pgm", x_buf, 0);
        WritePgm("out_0y.pgm", y_buf, 0);
        WritePgm("out_1x.pgm", x_buf, 1);
        WritePgm("out_1y.pgm", y_buf, 1);
        WritePgm("out_2x.pgm", x_buf, 2);
        WritePgm("out_2y.pgm", y_buf, 2);
        WritePgm("out_3x.pgm", x_buf, 3);
        WritePgm("out_3y.pgm", y_buf, 3);
        WritePgm("out_4x.pgm", x_buf, 4);
        WritePgm("out_4y.pgm", y_buf, 4);
    }

#if 0
    {
        std::cout << "\n<Evaluation binary LUT-Network>" << std::endl;

        // LUT-network
        auto layer_bl0 = bb::BinaryLutN<6, bb::Bit>::Create(layer_sl0->GetOutputShape());
        auto layer_bl1 = bb::BinaryLutN<6, bb::Bit>::Create(layer_sl1->GetOutputShape());
        auto layer_bl2 = bb::BinaryLutN<6, bb::Bit>::Create(layer_sl2->GetOutputShape());

        auto lut_net = bb::Sequential::Create();
        lut_net->Add(layer_bl0);
        lut_net->Add(layer_bl1);
        lut_net->Add(layer_bl2);

        // evaluation network
        auto eval_net = bb::Sequential::Create();
        eval_net->Add(bb::BinaryModulation<bb::Bit>::Create(lut_net, test_modulation_size));
        eval_net->Add(bb::Reduce<>::Create(td.t_shape));

        // set input shape
        eval_net->SetInputShape(td.x_shape);

        // テーブル化して取り込み(SetInputShape後に取り込みが必要)
        std::cout << "parameter copy to binary LUT-Network" << std::endl;
        layer_bl0->ImportLayer(layer_sl0);
        layer_bl1->ImportLayer(layer_sl1);
        layer_bl2->ImportLayer(layer_sl2);

        // evaluation
        if ( 1 ) {
            std::cout << "test_modulation_size  : " << test_modulation_size  << std::endl;
            bb::Runner<float>::create_t lut_runner_create;
            lut_runner_create.name           = "Lut_" + net_name;
            lut_runner_create.net            = eval_net;
            lut_runner_create.lossFunc       = bb::LossSoftmaxCrossEntropy<float>::Create();
            lut_runner_create.metricsFunc    = bb::MetricsCategoricalAccuracy<float>::Create();
            lut_runner_create.optimizer      = bb::OptimizerAdam<float>::Create();
            lut_runner_create.print_progress = true;
            auto lut_runner = bb::Runner<float>::Create(lut_runner_create);
            auto lut_accuracy = lut_runner->Evaluation(td, mini_batch_size);
            std::cout << "lut_accuracy : " << lut_accuracy << std::endl;
        }

        {
            // Verilog 出力
            std::string filename = "verilog/" + net_name + ".v";
            std::ofstream ofs(filename);
            ofs << "`timescale 1ns / 1ps\n\n";
            bb::ExportVerilog_LutLayers<>(ofs, net_name, lut_net);
            std::cout << "export : " << filename << "\n" << std::endl;

            // RTL simulation 用データの出力
            bb::WriteTestDataBinTextFile<float>("verilog/mnist_train.txt", "verilog/mnist_test.txt", td);
        }
    }
#endif
}


// end of file
