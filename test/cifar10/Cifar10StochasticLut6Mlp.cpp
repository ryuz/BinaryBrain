// --------------------------------------------------------------------------
//  BinaryBrain  -- binary network evaluation platform
//   CIFAR-10 sample
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
#include "bb/StochasticLut6.h"
#include "bb/BinaryLutN.h"
#include "bb/BatchNormalization.h"
#include "bb/ReLU.h"
#include "bb/LossSoftmaxCrossEntropy.h"
#include "bb/MetricsCategoricalAccuracy.h"
#include "bb/OptimizerAdam.h"
#include "bb/OptimizerSgd.h"
#include "bb/LoadCifar10.h"
#include "bb/ShuffleSet.h"
#include "bb/Utility.h"
#include "bb/Sequential.h"
#include "bb/Runner.h"
#include "bb/ExportVerilog.h"
#include "bb/UniformDistributionGenerator.h"


// MLP with LUT networks
void Cifar10StochasticLut6Mlp(int epoch_size, int mini_batch_size, int max_run_size, int lut_frame_mux_size, bool binary_mode, bool file_read)
{
    std::string net_name = "Cifar10StochasticLut6Mlp";
    int const mux_size = 7;

  // load cifar-10 data
#ifdef _DEBUG
    auto td = bb::LoadCifar10<>::Load(1);
    std::cout << "!!! debug mode !!!" << std::endl;
#else
    auto td = bb::LoadCifar10<>::Load();
#endif

    auto layer_sl0 = bb::StochasticLut6<>::Create({1024});
    auto layer_sl1 = bb::StochasticLut6<>::Create({360});
    auto layer_sl2 = bb::StochasticLut6<>::Create({60});
    auto layer_sl3 = bb::StochasticLut6<>::Create({10});

    {
        auto net = bb::Sequential::Create();
        net->Add(layer_sl0);
        net->Add(layer_sl1);
        net->Add(layer_sl2);
        net->Add(layer_sl3);
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
        auto layer_lut0 = bb::BinaryLutN<>::Create(layer_sl0->GetOutputShape());
        auto layer_lut1 = bb::BinaryLutN<>::Create(layer_sl1->GetOutputShape());
        auto layer_lut2 = bb::BinaryLutN<>::Create(layer_sl2->GetOutputShape());
        auto layer_lut3 = bb::BinaryLutN<>::Create(layer_sl3->GetOutputShape());

        auto lut_net = bb::Sequential::Create();
 //     lut_net->Add(bb::RealToBinary<float, bb::Bit>::Create(lut_frame_mux_size));
        lut_net->Add(bb::RealToBinary<float, bb::Bit>::Create(lut_frame_mux_size, bb::UniformDistributionGenerator<float>::Create(0.0f, 1.0f, 1)));
        lut_net->Add(layer_lut0);
        lut_net->Add(layer_lut1);
        lut_net->Add(layer_lut2);
        lut_net->Add(layer_lut3);
        lut_net->Add(bb::BinaryToReal<bb::Bit, float>::Create({10}, lut_frame_mux_size));
        lut_net->SetInputShape(td.x_shape);

        // テーブル化して取り込み(SetInputShape後に取り込みが必要)
        std::cout << "parameter copy to LUT-Network" << std::endl;
        layer_lut0->ImportLayer<float, float>(layer_sl0);
        layer_lut1->ImportLayer<float, float>(layer_sl1);
        layer_lut2->ImportLayer<float, float>(layer_sl2);
        layer_lut3->ImportLayer<float, float>(layer_sl3);

        // 評価
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

        {
            // Verilog 出力
            std::string filename = "verilog/" + net_name + ".v";
            std::ofstream ofs(filename);
            ofs << "`timescale 1ns / 1ps\n\n";
            bb::ExportVerilog_LutLayers<>(ofs, net_name, lut_net);
            std::cout << "export : " << filename << "\n" << std::endl;

            // RTL simulation 用データの出力
            bb::WriteTestDataBinTextFile<float>("verilog/cifar10_train.txt", "verilog/cifar10_test.txt", td);
        }
    }
}

// end of file
