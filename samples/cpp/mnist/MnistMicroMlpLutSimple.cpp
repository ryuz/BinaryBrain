// --------------------------------------------------------------------------
//  BinaryBrain  -- binary network evaluation platform
//   MNIST sample
//
//                                Copyright (C) 2018-2019 by Ryuji Fuchikami
// --------------------------------------------------------------------------


#include <iostream>

#include "bb/Sequential.h"
#include "bb/BinaryModulation.h"
#include "bb/Reduce.h"
#include "bb/MicroMlp.h"
#include "bb/BinaryLutN.h"
#include "bb/LossSoftmaxCrossEntropy.h"
#include "bb/MetricsCategoricalAccuracy.h"
#include "bb/OptimizerAdam.h"
#include "bb/Runner.h"
#include "bb/LoadMnist.h"
#include "bb/ExportVerilog.h"


void MnistMicroMlpLutSimple(int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read)
{
    std::string net_name       = "MnistMicroMlpLutSimple";
    std::string velilog_path   = "../../verilog/mnist/tb_mnist_lut_simple/";
    std::string velilog_module = "MnistLutSimple";

  // load MNIST data
#ifdef _DEBUG
    auto td = bb::LoadMnist<>::Load(64, 32);
    std::cout << "!!! debug mode !!!" << std::endl;
#else
    auto td = bb::LoadMnist<>::Load();
#endif

    auto layer_mm0 = bb::MicroMlp<6, 16, float>::Create(6*6*64);
    auto layer_mm1 = bb::MicroMlp<6, 16, float>::Create(6*64);
    auto layer_mm2 = bb::MicroMlp<6, 16, float>::Create(64);
    auto layer_mm3 = bb::MicroMlp<6, 16, float>::Create(6*6*10);
    auto layer_mm4 = bb::MicroMlp<6, 16, float>::Create(6*10);
    auto layer_mm5 = bb::MicroMlp<6, 16, float>::Create(10);

    {
        std::cout << "\n<Training>" << std::endl;

        // main network
        auto main_net = bb::Sequential::Create();
        main_net->Add(layer_mm0);
        main_net->Add(layer_mm1);
        main_net->Add(layer_mm2);
        main_net->Add(layer_mm3);
        main_net->Add(layer_mm4);
        main_net->Add(layer_mm5);

        // modulation wrapper
        auto net = bb::Sequential::Create();
        net->Add(bb::BinaryModulation<float>::Create(main_net, train_modulation_size, test_modulation_size));
        net->Add(bb::Reduce<float>::Create(td.t_shape));

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
        auto layer_bl0 = bb::BinaryLutN<>::Create(layer_mm0->GetOutputShape());
        auto layer_bl1 = bb::BinaryLutN<>::Create(layer_mm1->GetOutputShape());
        auto layer_bl2 = bb::BinaryLutN<>::Create(layer_mm2->GetOutputShape());
        auto layer_bl3 = bb::BinaryLutN<>::Create(layer_mm3->GetOutputShape());
        auto layer_bl4 = bb::BinaryLutN<>::Create(layer_mm4->GetOutputShape());
        auto layer_bl5 = bb::BinaryLutN<>::Create(layer_mm5->GetOutputShape());

        auto lut_net = bb::Sequential::Create();
        lut_net->Add(layer_bl0);
        lut_net->Add(layer_bl1);
        lut_net->Add(layer_bl2);
        lut_net->Add(layer_bl3);
        lut_net->Add(layer_bl4);
        lut_net->Add(layer_bl5);

        // evaluation network
        auto eval_net = bb::Sequential::Create();
        eval_net->Add(bb::BinaryModulation<bb::Bit>::Create(lut_net, test_modulation_size));
        eval_net->Add(bb::Reduce<>::Create(td.t_shape));

        // set input shape
        eval_net->SetInputShape(td.x_shape);

        // テーブル化して取り込み(SetInputShape後に取り込みが必要)
        std::cout << "parameter copy to binary LUT-Network" << std::endl;
        layer_bl0->ImportLayer(layer_mm0);
        layer_bl1->ImportLayer(layer_mm1);
        layer_bl2->ImportLayer(layer_mm2);
        layer_bl3->ImportLayer(layer_mm3);
        layer_bl4->ImportLayer(layer_mm4);
        layer_bl5->ImportLayer(layer_mm5);

        // 評価
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
            std::string filename = velilog_path + velilog_module + ".v";
            std::ofstream ofs(filename);
            ofs << "`timescale 1ns / 1ps\n\n";
            bb::ExportVerilog_LutModels(ofs, velilog_module, lut_net);
            std::cout << "export : " << filename << "\n" << std::endl;

            // RTL simulation 用データの出力
            bb::WriteTestDataBinTextFile<float>(velilog_path + "mnist_train.txt", velilog_path + "mnist_test.txt", td);
        }
    }
}


// end of file
