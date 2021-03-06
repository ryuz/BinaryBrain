﻿// --------------------------------------------------------------------------
//  BinaryBrain  -- binary network evaluation platform
//   CIFAR-10 sample
//
//                                Copyright (C) 2018-2019 by Ryuji Fuchikami
// --------------------------------------------------------------------------


#include <iostream>

#include "bb/Sequential.h"
#include "bb/StochasticLutN.h"
#include "bb/BinaryLutN.h"
#include "bb/BinaryModulation.h"
#include "bb/ShuffleModulation.h"
#include "bb/LossSoftmaxCrossEntropy.h"
#include "bb/MetricsCategoricalAccuracy.h"
#include "bb/OptimizerAdam.h"
#include "bb/Runner.h"
#include "bb/LoadCifar10.h"
#include "bb/ExportVerilog.h"



void Cifar10StochasticLutSimple(int epoch_size, int mini_batch_size, int test_modulation_size, bool binary_mode, bool file_read)
{
    std::string net_name       = "Cifar10StochasticLutSimple";
    std::string velilog_path   = "../../verilog/cifar10/";
    std::string velilog_module = "Cifar10LutSimple";

  // load cifar-10 data
#ifdef _DEBUG
    auto td = bb::LoadCifar10<>::Load(1);
    std::cout << "!!! debug mode !!!" << std::endl;
#else
    auto td = bb::LoadCifar10<>::Load();
#endif

    auto layer_sl0 = bb::StochasticLutN<6>::Create(3072);
    auto layer_sl1 = bb::StochasticLutN<6>::Create(512);
    auto layer_sl2 = bb::StochasticLutN<6>::Create(2160);
    auto layer_sl3 = bb::StochasticLutN<6>::Create(360);
    auto layer_sl4 = bb::StochasticLutN<6>::Create(60);
    auto layer_sl5 = bb::StochasticLutN<6>::Create(10);

    {
        std::cout << "\n<Training>" << std::endl;

        // create network
        auto net = bb::Sequential::Create();
        net->Add(layer_sl0);
        net->Add(layer_sl1);
        net->Add(layer_sl2);
        net->Add(layer_sl3);
        net->Add(layer_sl4);
        net->Add(layer_sl5);

        // set input shape
        net->SetInputShape(td.x_shape);

        // set binary mode
        net->SendCommand("binary false");
        if ( binary_mode ) {
            net->SendCommand("lut_binarize true");
        }
        else {
            net->SendCommand("lut_binarize false");
        }

        // print model information
        net->PrintInfo();

        std::cout << "-----------------------------------" << std::endl;
        std::cout << "epoch_size            : " << epoch_size            << std::endl;
        std::cout << "mini_batch_size       : " << mini_batch_size       << std::endl;
        std::cout << "lut_binarize          : " << binary_mode           << std::endl;
        std::cout << "file_read             : " << file_read             << std::endl;
        std::cout << "-----------------------------------" << std::endl;

        // fitting
        bb::Runner<float>::create_t runner_create;
        runner_create.name               = net_name;
        runner_create.net                = net;
        runner_create.lossFunc           = bb::LossSoftmaxCrossEntropy<float>::Create();
        runner_create.metricsFunc        = bb::MetricsCategoricalAccuracy<float>::Create();
        runner_create.optimizer          = bb::OptimizerAdam<float>::Create();
        runner_create.file_read          = file_read;       // 前の計算結果があれば読み込んで再開するか
        runner_create.file_write         = true;            // 計算結果をファイルに保存するか
        runner_create.print_progress     = true;            // 途中結果を表示
        runner_create.initial_evaluation = file_read;       // ファイルを読んだ場合は最初に評価しておく
        auto runner = bb::Runner<float>::Create(runner_create);
        runner->Fitting(td, epoch_size, mini_batch_size);
    }

    {
        std::cout << "\n<Evaluation binary LUT-Network>" << std::endl;
        
        // LUT-network
        auto layer_lut0 = bb::BinaryLutN<6, bb::Bit>::Create(layer_sl0->GetOutputShape());
        auto layer_lut1 = bb::BinaryLutN<6, bb::Bit>::Create(layer_sl1->GetOutputShape());
        auto layer_lut2 = bb::BinaryLutN<6, bb::Bit>::Create(layer_sl2->GetOutputShape());
        auto layer_lut3 = bb::BinaryLutN<6, bb::Bit>::Create(layer_sl3->GetOutputShape());
        auto layer_lut4 = bb::BinaryLutN<6, bb::Bit>::Create(layer_sl4->GetOutputShape());
        auto layer_lut5 = bb::BinaryLutN<6, bb::Bit>::Create(layer_sl5->GetOutputShape());

        auto lut_net = bb::Sequential::Create();
        lut_net->Add(bb::ShuffleModulation<bb::Bit>::Create(test_modulation_size, 1, 1));
        lut_net->Add(layer_lut0);
        lut_net->Add(bb::ShuffleModulation<bb::Bit>::Create(test_modulation_size, 1, 2));
        lut_net->Add(layer_lut1);
        lut_net->Add(bb::ShuffleModulation<bb::Bit>::Create(test_modulation_size, 1, 3));
        lut_net->Add(layer_lut2);
        lut_net->Add(bb::ShuffleModulation<bb::Bit>::Create(test_modulation_size, 1, 4));
        lut_net->Add(layer_lut3);
        lut_net->Add(bb::ShuffleModulation<bb::Bit>::Create(test_modulation_size, 1, 5));
        lut_net->Add(layer_lut4);
        lut_net->Add(bb::ShuffleModulation<bb::Bit>::Create(test_modulation_size, 1, 6));
        lut_net->Add(layer_lut5);

        // evaluation network
        auto eval_net = bb::BinaryModulation<bb::Bit>::Create(lut_net, test_modulation_size);

        // set input shape
        eval_net->SetInputShape(td.x_shape);

        // テーブル化して取り込み(SetInputShape後に取り込みが必要)
        std::cout << "parameter copy to binary LUT-Network" << std::endl;
        layer_lut0->ImportLayer(layer_sl0);
        layer_lut1->ImportLayer(layer_sl1);
        layer_lut2->ImportLayer(layer_sl2);
        layer_lut3->ImportLayer(layer_sl3);
        layer_lut4->ImportLayer(layer_sl4);
        layer_lut5->ImportLayer(layer_sl5);

        if ( 1 ) {
            // 評価
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
            std::string filename = velilog_path + velilog_module + ".v";
            std::ofstream ofs(filename);
            ofs << "`timescale 1ns / 1ps\n\n";
            bb::ExportVerilog_LutModels(ofs, net_name, lut_net);
            std::cout << "export : " << velilog_module << "\n" << std::endl;

            // RTL simulation 用データの出力
            bb::WriteTestDataBinTextFile<float>(velilog_path + "cifar10_train.txt", velilog_path + "cifar10_test.txt", td);
        }
    }
}


// end of file
