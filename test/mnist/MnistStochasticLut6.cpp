// --------------------------------------------------------------------------
//  BinaryBrain  -- binary network evaluation platform
//   MNIST sample
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
// --------------------------------------------------------------------------


#include <iostream>
#include <fstream>
#include <numeric>
#include <random>
#include <chrono>

#include "bb/RealToBinary.h"
#include "bb/BinaryToReal.h"
#include "bb/BinaryLutN.h"
#include "bb/MicroMlp.h"
#include "bb/StochasticLut6.h"
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
void MnistStochasticLut6(int epoch_size, size_t mini_batch_size, bool binary_mode)
{
  std::string net_name = "MnistStochasticLut6";

  // load MNIST data
#ifdef _DEBUG
	auto td = bb::LoadMnist<>::Load(10, 1024, 1024);
#else
    auto td = bb::LoadMnist<>::Load(10);
#endif

    auto layer_sl0 = bb::StochasticLut6<>::Create({360});
    auto layer_sl1 = bb::StochasticLut6<>::Create({60});
    auto layer_sl2 = bb::StochasticLut6<>::Create({10});

    {
        auto net = bb::Sequential::Create();
        net->Add(layer_sl0);
        net->Add(layer_sl1);
        net->Add(layer_sl2);
        net->SetInputShape(td.x_shape);
    
        net->PrintInfo();

        bb::Runner<float>::create_t runner_create;
        runner_create.name        = net_name;
        runner_create.net         = net;
        runner_create.lossFunc    = bb::LossSoftmaxCrossEntropy<float>::Create();
        runner_create.metricsFunc = bb::MetricsCategoricalAccuracy<float>::Create();
    //  runner_create.optimizer   = bb::OptimizerSgd<float>::Create(0.01f);
        runner_create.optimizer   = bb::OptimizerAdam<float>::Create();
        runner_create.print_progress = true;
        runner_create.file_write = true;
        runner_create.initial_evaluation = false;
        auto runner = bb::Runner<float>::Create(runner_create);
    
        runner->Fitting(td, epoch_size, mini_batch_size);
    }

    {
        int frame_mux_size = 15;

        // LUT-network
        auto layer_lut0 = bb::BinaryLutN<>::Create(layer_sl0->GetOutputShape());
        auto layer_lut1 = bb::BinaryLutN<>::Create(layer_sl1->GetOutputShape());
        auto layer_lut2 = bb::BinaryLutN<>::Create(layer_sl2->GetOutputShape());

        auto lut_net = bb::Sequential::Create();
        lut_net->Add(bb::RealToBinary<float, bb::Bit>::Create(frame_mux_size));
        lut_net->Add(layer_lut0);
        lut_net->Add(layer_lut1);
        lut_net->Add(layer_lut2);
        lut_net->Add(bb::BinaryToReal<bb::Bit, float>::Create({10}, frame_mux_size));
        lut_net->SetInputShape(td.x_shape);

        // テーブル化して取り込み(SetInputShape後に取り込みが必要)
        layer_lut0->ImportLayer<float, float>(layer_sl0);
        layer_lut1->ImportLayer<float, float>(layer_sl1);
        layer_lut2->ImportLayer<float, float>(layer_sl2);

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
        }
    }
}

