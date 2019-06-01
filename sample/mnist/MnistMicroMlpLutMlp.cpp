﻿// --------------------------------------------------------------------------
//  BinaryBrain  -- binary network evaluation platform
//   MNIST sample
//
//                                Copyright (C) 2018-2019 by Ryuji Fuchikami
// --------------------------------------------------------------------------


#include <iostream>
#include <fstream>
#include <numeric>
#include <random>
#include <chrono>

#include "bb/RealToBinary.h"
#include "bb/BinaryToReal.h"
#include "bb/Reduce.h"
#include "bb/MicroMlp.h"
#include "bb/BinaryLutN.h"
#include "bb/BatchNormalization.h"
#include "bb/ReLU.h"
#include "bb/LossSoftmaxCrossEntropy.h"
#include "bb/MetricsCategoricalAccuracy.h"
#include "bb/OptimizerAdam.h"
#include "bb/OptimizerSgd.h"
#include "bb/LoadMnist.h"
#include "bb/ShuffleSet.h"
#include "bb/Utility.h"
#include "bb/Sequential.h"
#include "bb/Runner.h"
#include "bb/ExportVerilog.h"


// MNIST CNN with LUT networks
void MnistMicroMlpLutMlp(int epoch_size, int mini_batch_size, int max_run_size, int frame_mux_size, int lut_frame_mux_size, bool binary_mode, bool file_read)
{
    std::string net_name = "MnistMicroMlpLutMlp";

  // load MNIST data
#ifdef _DEBUG
    auto td = bb::LoadMnist<>::Load(10, 64, 32);
    std::cout << "!!! debug mode !!!" << std::endl;
#else
    auto td = bb::LoadMnist<>::Load(10);
#endif

    auto layer_mm0 = bb::MicroMlp<>::Create(1024);
    auto layer_mm1 = bb::MicroMlp<>::Create(480);
    auto layer_mm2 = bb::MicroMlp<>::Create(70);

    {
        auto net = bb::Sequential::Create();
        net->Add(bb::RealToBinary<>::Create(frame_mux_size));
        net->Add(layer_mm0);
        net->Add(layer_mm1);
        net->Add(layer_mm2);
        net->Add(bb::BinaryToReal<>::Create(frame_mux_size));
        net->Add(bb::Reduce<>::Create(td.t_shape));
        net->SetInputShape(td.x_shape);

        if ( binary_mode ) {
            net->SendCommand("binary true");
            std::cout << "binary mode" << std::endl;
        }

        net->PrintInfo();

        // fitting
        bb::Runner<float>::create_t runner_create;
        runner_create.name               = net_name;
        runner_create.net                = net;
        runner_create.lossFunc           = bb::LossSoftmaxCrossEntropy<float>::Create();
        runner_create.metricsFunc        = bb::MetricsCategoricalAccuracy<float>::Create();
        runner_create.optimizer          = bb::OptimizerAdam<float>::Create();
        runner_create.max_run_size       = max_run_size;    // 実際の1回の実行サイズ
        runner_create.file_read          = file_read;       // 前の計算結果があれば読み込んで再開するか
        runner_create.file_write         = true;            // 計算結果をファイルに保存するか
        runner_create.print_progress     = true;            // 途中結果を表示
        runner_create.initial_evaluation = file_read;       // ファイルを読んだ場合は最初に評価しておく
        auto runner = bb::Runner<float>::Create(runner_create);
        runner->Fitting(td, epoch_size, mini_batch_size);
    }

    {
        // LUT-network
        auto layer_lut0 = bb::BinaryLutN<>::Create(layer_mm0->GetOutputShape());
        auto layer_lut1 = bb::BinaryLutN<>::Create(layer_mm1->GetOutputShape());
        auto layer_lut2 = bb::BinaryLutN<>::Create(layer_mm2->GetOutputShape());

        auto lut_net = bb::Sequential::Create();
        lut_net->Add(bb::RealToBinary<bb::Bit>::Create(lut_frame_mux_size));
        lut_net->Add(layer_lut0);
        lut_net->Add(layer_lut1);
        lut_net->Add(layer_lut2);
        lut_net->Add(bb::BinaryToReal<bb::Bit>::Create(lut_frame_mux_size));
        lut_net->Add(bb::Reduce<>::Create(td.t_shape));
        lut_net->SetInputShape(td.x_shape);

        // テーブル化して取り込み(SetInputShape後に取り込みが必要)
        std::cout << "parameter copy to LUT-Network" << std::endl;
        layer_lut0->ImportLayer(layer_mm0);
        layer_lut1->ImportLayer(layer_mm1);
        layer_lut2->ImportLayer(layer_mm2);

        // 評価
        if ( 1 ) {
            bb::Runner<float>::create_t lut_runner_create;
            lut_runner_create.name           = "Lut_" + net_name;
            lut_runner_create.net            = lut_net;
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
}


// end of file
