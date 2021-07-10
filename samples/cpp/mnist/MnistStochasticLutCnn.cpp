// --------------------------------------------------------------------------
//  BinaryBrain  -- binary network evaluation platform
//   MNIST sample
//
//                                Copyright (C) 2018-2019 by Ryuji Fuchikami
// --------------------------------------------------------------------------


#include <iostream>

#include "bb/Sequential.h"
#include "bb/BinaryModulation.h"
#include "bb/StochasticLutN.h"
#include "bb/StochasticMaxPooling2x2.h"
#include "bb/BinaryLutN.h"
#include "bb/Convolution2d.h"
#include "bb/MaxPooling.h"
#include "bb/LossSoftmaxCrossEntropy.h"
#include "bb/MetricsCategoricalAccuracy.h"
#include "bb/OptimizerAdam.h"
#include "bb/Runner.h"
#include "bb/LoadMnist.h"
#include "bb/ExportVerilog.h"



void MnistStochasticLutCnn(int epoch_size, int mini_batch_size, int test_modulation_size, bool binary_mode, bool file_read)
{
    std::string net_name       = "MnistStochasticLutCnn";
    std::string velilog_path   = "../../verilog/mnist/";
    std::string velilog_module = "MnistLutCnn";

  // load MNIST data
#ifdef _DEBUG
    auto td = bb::LoadMnist<>::Load(512, 128);
    std::cout << "!!! debug mode !!!" << std::endl;
#else
    auto td = bb::LoadMnist<>::Load();
#endif

    // create network
    auto layer_cnv0_sl0 = bb::StochasticLutN<6>::Create(6*36);
    auto layer_cnv0_sl1 = bb::StochasticLutN<6>::Create(36);
    auto layer_cnv1_sl0 = bb::StochasticLutN<6>::Create(6*2*36);
    auto layer_cnv1_sl1 = bb::StochasticLutN<6>::Create(2*36);
    auto layer_cnv2_sl0 = bb::StochasticLutN<6>::Create(6*2*36);
    auto layer_cnv2_sl1 = bb::StochasticLutN<6>::Create(2*36);
    auto layer_cnv3_sl0 = bb::StochasticLutN<6>::Create(6*4*36);
    auto layer_cnv3_sl1 = bb::StochasticLutN<6>::Create(4*36);
    auto layer_sl4      = bb::StochasticLutN<6>::Create(6*128);
    auto layer_sl5      = bb::StochasticLutN<6>::Create(128);
    auto layer_sl6      = bb::StochasticLutN<6>::Create(6*6*10);
    auto layer_sl7      = bb::StochasticLutN<6>::Create(6*10);
    auto layer_sl8      = bb::StochasticLutN<6>::Create(10);

    {
        std::cout << "\n<Training>" << std::endl;
        
        auto cnv0_sub = bb::Sequential::Create();
        cnv0_sub->Add(layer_cnv0_sl0);
        cnv0_sub->Add(layer_cnv0_sl1);
        auto cnv0 = bb::Convolution2d<>::Create(cnv0_sub, 3, 3);

        auto cnv1_sub = bb::Sequential::Create();
        cnv1_sub->Add(layer_cnv1_sl0);
        cnv1_sub->Add(layer_cnv1_sl1);
        auto cnv1 = bb::Convolution2d<>::Create(cnv1_sub, 3, 3);

        auto pol0 = bb::StochasticMaxPooling2x2<float, float>::Create();

        auto cnv2_sub = bb::Sequential::Create();
        cnv2_sub->Add(layer_cnv2_sl0);
        cnv2_sub->Add(layer_cnv2_sl1);
        auto cnv2 = bb::Convolution2d<>::Create(cnv2_sub, 3, 3);

        auto cnv3_sub = bb::Sequential::Create();
        cnv3_sub->Add(layer_cnv3_sl0);
        cnv3_sub->Add(layer_cnv3_sl1);
        auto cnv3 = bb::Convolution2d<>::Create(cnv3_sub, 3, 3);

        auto pol1 = bb::StochasticMaxPooling2x2<float, float>::Create();

        auto cnv4_sub = bb::Sequential::Create();
        cnv4_sub->Add(layer_sl4);
        cnv4_sub->Add(layer_sl5);
        cnv4_sub->Add(layer_sl6);
        cnv4_sub->Add(layer_sl7);
        cnv4_sub->Add(layer_sl8);
        auto cnv4 = bb::Convolution2d<>::Create(cnv4_sub, 4, 4);


        auto net = bb::Sequential::Create();
        net->Add(cnv0);
        net->Add(cnv1);
        net->Add(pol0);
        net->Add(cnv2);
        net->Add(cnv3);
        net->Add(pol1);
        net->Add(cnv4);

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

        // run fitting
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
    
        {
            // Verilog 出力
            std::vector< std::shared_ptr< bb::Filter2d > >  vec_cnv0;
            std::vector< std::shared_ptr< bb::Filter2d > >  vec_cnv1;
            std::vector< std::shared_ptr< bb::Filter2d > >  vec_cnv2;

            vec_cnv0.push_back(cnv0);
            vec_cnv0.push_back(cnv1);
            vec_cnv0.push_back(pol0);
            vec_cnv1.push_back(cnv2);
            vec_cnv1.push_back(cnv3);
            vec_cnv1.push_back(pol1);
            vec_cnv2.push_back(cnv4);

            std::string filename = velilog_path + velilog_module + ".v";
            std::ofstream ofs(filename);
            ofs << "`timescale 1ns / 1ps\n\n";
            bb::ExportVerilog_LutCnnLayersAxi4s(ofs, velilog_module + "Cnv0", vec_cnv0);
            bb::ExportVerilog_LutCnnLayersAxi4s(ofs, velilog_module + "Cnv1", vec_cnv1);
            bb::ExportVerilog_LutCnnLayersAxi4s(ofs, velilog_module + "Cnv2", vec_cnv2);
            std::cout << "export : " << filename << "\n" << std::endl;
        }
    }


    // ここまでで完結しているが、普通のFPGA的なLUTモデルも用意しているので、そちらに
    // コピーしての推論も実行してみる
    {
        std::cout << "\n<Evaluation binary LUT-Network>" << std::endl;

        // LUT-network
        auto layer_cnv0_lut0 = bb::BinaryLutN<>::Create(layer_cnv0_sl0->GetOutputShape());
        auto layer_cnv0_lut1 = bb::BinaryLutN<>::Create(layer_cnv0_sl1->GetOutputShape());
        auto layer_cnv1_lut0 = bb::BinaryLutN<>::Create(layer_cnv1_sl0->GetOutputShape());
        auto layer_cnv1_lut1 = bb::BinaryLutN<>::Create(layer_cnv1_sl1->GetOutputShape());
        auto layer_cnv2_lut0 = bb::BinaryLutN<>::Create(layer_cnv2_sl0->GetOutputShape());
        auto layer_cnv2_lut1 = bb::BinaryLutN<>::Create(layer_cnv2_sl1->GetOutputShape());
        auto layer_cnv3_lut0 = bb::BinaryLutN<>::Create(layer_cnv3_sl0->GetOutputShape());
        auto layer_cnv3_lut1 = bb::BinaryLutN<>::Create(layer_cnv3_sl1->GetOutputShape());
        auto layer_lut4      = bb::BinaryLutN<>::Create(layer_sl4->GetOutputShape());
        auto layer_lut5      = bb::BinaryLutN<>::Create(layer_sl5->GetOutputShape());
        auto layer_lut6      = bb::BinaryLutN<>::Create(layer_sl6->GetOutputShape());
        auto layer_lut7      = bb::BinaryLutN<>::Create(layer_sl7->GetOutputShape());
        auto layer_lut8      = bb::BinaryLutN<>::Create(layer_sl8->GetOutputShape());

        auto cnv0_sub = bb::Sequential::Create();
        cnv0_sub->Add(layer_cnv0_lut0);
        cnv0_sub->Add(layer_cnv0_lut1);

        auto cnv1_sub = bb::Sequential::Create();
        cnv1_sub->Add(layer_cnv1_lut0);
        cnv1_sub->Add(layer_cnv1_lut1);

        auto cnv2_sub = bb::Sequential::Create();
        cnv2_sub->Add(layer_cnv2_lut0);
        cnv2_sub->Add(layer_cnv2_lut1);

        auto cnv3_sub = bb::Sequential::Create();
        cnv3_sub->Add(layer_cnv3_lut0);
        cnv3_sub->Add(layer_cnv3_lut1);

        auto cnv4_sub = bb::Sequential::Create();
        cnv4_sub->Add(layer_lut4);
        cnv4_sub->Add(layer_lut5);
        cnv4_sub->Add(layer_lut6);
        cnv4_sub->Add(layer_lut7);
        cnv4_sub->Add(layer_lut8);

        auto cnv0 = bb::Convolution2d<bb::Bit>::Create(cnv0_sub, 3, 3);
        auto cnv1 = bb::Convolution2d<bb::Bit>::Create(cnv1_sub, 3, 3);
        auto pol0 = bb::MaxPooling<bb::Bit>::Create(2, 2);

        auto cnv2 = bb::Convolution2d<bb::Bit>::Create(cnv2_sub, 3, 3);
        auto cnv3 = bb::Convolution2d<bb::Bit>::Create(cnv3_sub, 3, 3);
        auto pol1 = bb::MaxPooling<bb::Bit>::Create(2, 2);

        // 28x28 以外も入力できるように最終段も畳み込みに変換
        auto cnv4 = bb::Convolution2d<bb::Bit>::Create(cnv4_sub, 4, 4);

        auto lut_net = bb::Sequential::Create();
        lut_net->Add(cnv0);
        lut_net->Add(cnv1);
        lut_net->Add(pol0);
        lut_net->Add(cnv2);
        lut_net->Add(cnv3);
        lut_net->Add(pol1);
        lut_net->Add(cnv4);

        // evaluation network
        auto eval_net = bb::BinaryModulation<bb::Bit>::Create(lut_net, test_modulation_size);

        // set input shape
        eval_net->SetInputShape(td.x_shape);

        // テーブル化して取り込み(現状まだSetInputShape後の取り込みが必要)
        std::cout << "parameter copy to binary LUT-Network" << std::endl;
        layer_cnv0_lut0->ImportLayer(layer_cnv0_sl0);
        layer_cnv0_lut1->ImportLayer(layer_cnv0_sl1);
        layer_cnv1_lut0->ImportLayer(layer_cnv1_sl0);
        layer_cnv1_lut1->ImportLayer(layer_cnv1_sl1);
        layer_cnv2_lut0->ImportLayer(layer_cnv2_sl0);
        layer_cnv2_lut1->ImportLayer(layer_cnv2_sl1);
        layer_cnv3_lut0->ImportLayer(layer_cnv3_sl0);
        layer_cnv3_lut1->ImportLayer(layer_cnv3_sl1);
        layer_lut4     ->ImportLayer(layer_sl4);
        layer_lut5     ->ImportLayer(layer_sl5);
        layer_lut6     ->ImportLayer(layer_sl6);
        layer_lut7     ->ImportLayer(layer_sl7);
        layer_lut8     ->ImportLayer(layer_sl8);

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
            std::vector< std::shared_ptr< bb::Filter2d > >  vec_cnv0;
            std::vector< std::shared_ptr< bb::Filter2d > >  vec_cnv1;
            std::vector< std::shared_ptr< bb::Filter2d > >  vec_cnv2;

            vec_cnv0.push_back(cnv0);
            vec_cnv0.push_back(cnv1);
            vec_cnv0.push_back(pol0);
            vec_cnv1.push_back(cnv2);
            vec_cnv1.push_back(cnv3);
            vec_cnv1.push_back(pol1);
            vec_cnv2.push_back(cnv4);

            std::string filename = velilog_path + velilog_module + "_2.v";
            std::ofstream ofs(filename);
            ofs << "`timescale 1ns / 1ps\n\n";
            bb::ExportVerilog_LutCnnLayersAxi4s(ofs, velilog_module + "Cnv0", vec_cnv0);
            bb::ExportVerilog_LutCnnLayersAxi4s(ofs, velilog_module + "Cnv1", vec_cnv1);
            bb::ExportVerilog_LutCnnLayersAxi4s(ofs, velilog_module + "Cnv2", vec_cnv2);
            std::cout << "export : " << filename << "\n" << std::endl;
            
            // write test image
            bb::WriteTestDataImage<float>(velilog_path + "mnist_test_160x120.ppm", 160, 120, td);
            bb::WriteTestDataImage<float>(velilog_path + "mnist_test_640x480.ppm", 640, 480, td);
        }
    }
}


// end of file
