// --------------------------------------------------------------------------
//  BinaryBrain  -- binary network evaluation platform
//   CIFAR-10 sample
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
#include "bb/LossSoftmaxCrossEntropy.h"
#include "bb/MetricsCategoricalAccuracy.h"
#include "bb/Runner.h"
#include "bb/LoadCifar10.h"
#include "bb/ExportVerilog.h"


void Cifar10SparseLutMlp(int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read)
{
    std::string net_name = "Cifar10SparseLutMlp";
     
  // load cifar-10 data
#ifdef _DEBUG
    auto td = bb::LoadCifar10<>::Load(1);
    std::cout << "!!! debug mode !!!" << std::endl;
#else
    auto td = bb::LoadCifar10<>::Load();
#endif

     // create network
    auto layer_sl0 = bb::SparseLutN<6, float>::Create(3072);
    auto layer_sl1 = bb::SparseLutN<6, float>::Create(512);
    auto layer_sl2 = bb::SparseLutN<6, float>::Create(2160);
    auto layer_sl3 = bb::SparseLutN<6, float>::Create(360);
    auto layer_sl4 = bb::SparseLutN<6, float>::Create(60);
    auto layer_sl5 = bb::SparseLutN<6, float>::Create(10);

    {
        std::cout << "\n<Training>" << std::endl;
        
        auto main_net = bb::Sequential::Create();
        main_net->Add(layer_sl0);
        main_net->Add(layer_sl1);
        main_net->Add(layer_sl2);
        main_net->Add(layer_sl3);
        main_net->Add(layer_sl4);
        main_net->Add(layer_sl5);

        // modulation wrapper
        auto net = bb::BinaryModulation<float>::Create(main_net, train_modulation_size, test_modulation_size);

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
        auto layer_bl0 = bb::BinaryLutN<6, bb::Bit>::Create(layer_sl0->GetOutputShape());
        auto layer_bl1 = bb::BinaryLutN<6, bb::Bit>::Create(layer_sl1->GetOutputShape());
        auto layer_bl2 = bb::BinaryLutN<6, bb::Bit>::Create(layer_sl2->GetOutputShape());
        auto layer_bl3 = bb::BinaryLutN<6, bb::Bit>::Create(layer_sl3->GetOutputShape());
        auto layer_bl4 = bb::BinaryLutN<6, bb::Bit>::Create(layer_sl4->GetOutputShape());
        auto layer_bl5 = bb::BinaryLutN<6, bb::Bit>::Create(layer_sl5->GetOutputShape());

        auto lut_net = bb::Sequential::Create();
        lut_net->Add(layer_bl0);
        lut_net->Add(layer_bl1);
        lut_net->Add(layer_bl2);
        lut_net->Add(layer_bl3);
        lut_net->Add(layer_bl4);
        lut_net->Add(layer_bl5);

        // evaluation network
        auto eval_net = bb::BinaryModulation<bb::Bit>::Create(lut_net, test_modulation_size);

        // set input shape
        eval_net->SetInputShape(td.x_shape);


        // テーブル化して取り込み(現状まだSetInputShape後の取り込みが必要)
        std::cout << "parameter copy to binary LUT-Network" << std::endl;
        layer_bl0->ImportLayer(layer_sl0);
        layer_bl1->ImportLayer(layer_sl1);
        layer_bl2->ImportLayer(layer_sl2);
        layer_bl3->ImportLayer(layer_sl3);
        layer_bl4->ImportLayer(layer_sl4);
        layer_bl5->ImportLayer(layer_sl5);

        // 評価
        if ( 1 ) {
            std::cout << "test_modulation_size  : " << test_modulation_size  << std::endl;
            bb::Runner<float>::create_t lut_runner_create;
            lut_runner_create.name        = "Lut_" + net_name;
            lut_runner_create.net         = eval_net;
            lut_runner_create.lossFunc    = bb::LossSoftmaxCrossEntropy<float>::Create();
            lut_runner_create.metricsFunc = bb::MetricsCategoricalAccuracy<float>::Create();
            lut_runner_create.optimizer   = bb::OptimizerAdam<float>::Create();
            lut_runner_create.initial_evaluation = false;
            lut_runner_create.print_progress = true;    // 途中結果を出力
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
