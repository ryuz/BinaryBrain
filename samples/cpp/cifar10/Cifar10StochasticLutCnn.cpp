// --------------------------------------------------------------------------
//  BinaryBrain  -- binary network evaluation platform
//   CIFAR-10 sample
//
//                                Copyright (C) 2018-2019 by Ryuji Fuchikami
// --------------------------------------------------------------------------


#include <iostream>

#include "bb/Sequential.h"
#include "bb/BinaryModulation.h"
#include "bb/ShuffleModulation.h"
#include "bb/StochasticLutN.h"
#include "bb/BinaryLutN.h"
#include "bb/Convolution2d.h"
#include "bb/BatchNormalization.h"
#include "bb/BackpropagatedBatchNormalization.h"
#include "bb/MaxPooling.h"
#include "bb/LossSoftmaxCrossEntropy.h"
#include "bb/MetricsCategoricalAccuracy.h"
#include "bb/OptimizerAdam.h"
#include "bb/Runner.h"
#include "bb/LoadCifar10.h"
#include "bb/ExportVerilog.h"


void Cifar10StochasticLutCnn(int epoch_size, int mini_batch_size, int test_modulation_size, bool binary_mode, bool file_read)
{
    std::string net_name = "Cifar10StochasticLutCnn";

  // load cifar-10 data
#ifdef _DEBUG
    auto td = bb::LoadCifar10<>::Load(1);
    std::cout << "!!! debug mode !!!" << std::endl;
#else
    auto td = bb::LoadCifar10<>::Load();
#endif

    // create network
    auto layer_cnv0_sl0 = bb::StochasticLutN<6>::Create(192);
    auto layer_cnv0_sl1 = bb::StochasticLutN<6>::Create(32);
    
    auto layer_cnv1_sl0 = bb::StochasticLutN<6>::Create(192);
    auto layer_cnv1_sl1 = bb::StochasticLutN<6>::Create(32);

    auto layer_cnv2_sl0 = bb::StochasticLutN<6>::Create(384);
    auto layer_cnv2_sl1 = bb::StochasticLutN<6>::Create(64);
    
    auto layer_cnv3_sl0 = bb::StochasticLutN<6>::Create(384);
    auto layer_cnv3_sl1 = bb::StochasticLutN<6>::Create(64);
    
    auto layer_sl4      = bb::StochasticLutN<6>::Create(360);
    auto layer_sl5      = bb::StochasticLutN<6>::Create(60);
    auto layer_sl6      = bb::StochasticLutN<6>::Create(10);

    {
        std::cout << "\n<Training>" << std::endl;

        auto cnv0_sub = bb::Sequential::Create();
        cnv0_sub->Add(layer_cnv0_sl0);
        cnv0_sub->Add(layer_cnv0_sl1);

        auto cnv1_sub = bb::Sequential::Create();
        cnv1_sub->Add(layer_cnv1_sl0);
        cnv1_sub->Add(layer_cnv1_sl1);

        auto cnv2_sub = bb::Sequential::Create();
        cnv2_sub->Add(layer_cnv2_sl0);
        cnv2_sub->Add(layer_cnv2_sl1);

        auto cnv3_sub = bb::Sequential::Create();
        cnv3_sub->Add(layer_cnv3_sl0);
        cnv3_sub->Add(layer_cnv3_sl1);
        
        auto net = bb::Sequential::Create();
        net->Add(bb::Convolution2d<>::Create(cnv0_sub, 3, 3));
        net->Add(bb::Convolution2d<>::Create(cnv1_sub, 3, 3));
        net->Add(bb::MaxPooling<>::Create(2, 2));
        net->Add(bb::BackpropagatedBatchNormalization<>::Create());
        net->Add(bb::Convolution2d<>::Create(cnv2_sub, 3, 3));
        net->Add(bb::Convolution2d<>::Create(cnv3_sub, 3, 3));
        net->Add(bb::MaxPooling<>::Create(2, 2));
        net->Add(bb::BackpropagatedBatchNormalization<>::Create());
        net->Add(layer_sl4);
        net->Add(layer_sl5);
        net->Add(layer_sl6);

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
        runner_create.initial_evaluation = false;//file_read;       // ファイルを読んだ場合は最初に評価しておく
        auto runner = bb::Runner<float>::Create(runner_create);
        runner->Fitting(td, epoch_size, mini_batch_size);
    }


    {
         std::cout << "\n<Evaluation binary LUT-Network>" << std::endl;

        // LUT-network
        auto layer_cnv0_bl0 = bb::BinaryLutN<>::Create(layer_cnv0_sl0->GetOutputShape());
        auto layer_cnv0_bl1 = bb::BinaryLutN<>::Create(layer_cnv0_sl1->GetOutputShape());
        auto layer_cnv1_bl0 = bb::BinaryLutN<>::Create(layer_cnv1_sl0->GetOutputShape());
        auto layer_cnv1_bl1 = bb::BinaryLutN<>::Create(layer_cnv1_sl1->GetOutputShape());
        auto layer_cnv2_bl0 = bb::BinaryLutN<>::Create(layer_cnv2_sl0->GetOutputShape());
        auto layer_cnv2_bl1 = bb::BinaryLutN<>::Create(layer_cnv2_sl1->GetOutputShape());
        auto layer_cnv3_bl0 = bb::BinaryLutN<>::Create(layer_cnv3_sl0->GetOutputShape());
        auto layer_cnv3_bl1 = bb::BinaryLutN<>::Create(layer_cnv3_sl1->GetOutputShape());
        auto layer_bl4      = bb::BinaryLutN<>::Create(layer_sl4->GetOutputShape());
        auto layer_bl5      = bb::BinaryLutN<>::Create(layer_sl5->GetOutputShape());
        auto layer_bl6      = bb::BinaryLutN<>::Create(layer_sl6->GetOutputShape());

        auto cnv0_sub = bb::Sequential::Create();
        cnv0_sub->Add(bb::ShuffleModulation<>::Create(test_modulation_size, 30*30, 1));
        cnv0_sub->Add(layer_cnv0_bl0);
        cnv0_sub->Add(bb::ShuffleModulation<>::Create(test_modulation_size, 30*30, 2));
        cnv0_sub->Add(layer_cnv0_bl1);

        auto cnv1_sub = bb::Sequential::Create();
        cnv1_sub->Add(bb::ShuffleModulation<>::Create(test_modulation_size, 28*28, 3));
        cnv1_sub->Add(layer_cnv1_bl0);
        cnv1_sub->Add(bb::ShuffleModulation<>::Create(test_modulation_size, 28*28, 4));
        cnv1_sub->Add(layer_cnv1_bl1);

        auto cnv2_sub = bb::Sequential::Create();
        cnv2_sub->Add(bb::ShuffleModulation<>::Create(test_modulation_size, 12*12, 5));
        cnv2_sub->Add(layer_cnv2_bl0);
        cnv2_sub->Add(bb::ShuffleModulation<>::Create(test_modulation_size, 12*12, 6));
        cnv2_sub->Add(layer_cnv2_bl1);

        auto cnv3_sub = bb::Sequential::Create();
        cnv3_sub->Add(bb::ShuffleModulation<>::Create(test_modulation_size, 10*10, 7));
        cnv3_sub->Add(layer_cnv3_bl0);
        cnv3_sub->Add(bb::ShuffleModulation<>::Create(test_modulation_size, 10*10, 8));
        cnv3_sub->Add(layer_cnv3_bl1);

        auto cnv4_sub = bb::Sequential::Create();
        cnv4_sub->Add(bb::ShuffleModulation<>::Create(test_modulation_size, 1*1, 9));
        cnv4_sub->Add(layer_bl4);
        cnv4_sub->Add(bb::ShuffleModulation<>::Create(test_modulation_size, 1*1, 10));
        cnv4_sub->Add(layer_bl5);
        cnv4_sub->Add(bb::ShuffleModulation<>::Create(test_modulation_size, 1*1, 11));
        cnv4_sub->Add(layer_bl6);

        auto cnv0 = bb::Convolution2d<bb::Bit>::Create(cnv0_sub, 3, 3);
        auto cnv1 = bb::Convolution2d<bb::Bit>::Create(cnv1_sub, 3, 3);
        auto pol0 = bb::MaxPooling<bb::Bit>::Create(2, 2);

        auto cnv2 = bb::Convolution2d<bb::Bit>::Create(cnv2_sub, 3, 3);
        auto cnv3 = bb::Convolution2d<bb::Bit>::Create(cnv3_sub, 3, 3);
        auto pol1 = bb::MaxPooling<bb::Bit>::Create(2, 2);

        auto cnv4 = bb::Convolution2d<bb::Bit>::Create(cnv4_sub, 5, 5);

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
        std::cout << "parameter copy to LUT-Network" << std::endl;
        layer_cnv0_bl0->ImportLayer(layer_cnv0_sl0);
        layer_cnv0_bl1->ImportLayer(layer_cnv0_sl1);
        layer_cnv1_bl0->ImportLayer(layer_cnv1_sl0);
        layer_cnv1_bl1->ImportLayer(layer_cnv1_sl1);
        layer_cnv2_bl0->ImportLayer(layer_cnv2_sl0);
        layer_cnv2_bl1->ImportLayer(layer_cnv2_sl1);
        layer_cnv3_bl0->ImportLayer(layer_cnv3_sl0);
        layer_cnv3_bl1->ImportLayer(layer_cnv3_sl1);
        layer_bl4     ->ImportLayer(layer_sl4);
        layer_bl5     ->ImportLayer(layer_sl5);
        layer_bl6     ->ImportLayer(layer_sl6);

        if ( 1 ) {
            // 評価
            std::cout << "test_modulation_size : " << test_modulation_size << std::endl;

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

            std::string filename = "verilog/" + net_name + ".v";
            std::ofstream ofs(filename);
            ofs << "`timescale 1ns / 1ps\n\n";
            bb::ExportVerilog_LutCnnLayersAxi4s(ofs, net_name + "Cnv0", vec_cnv0);
            bb::ExportVerilog_LutCnnLayersAxi4s(ofs, net_name + "Cnv1", vec_cnv1);
            bb::ExportVerilog_LutCnnLayersAxi4s(ofs, net_name + "Cnv2", vec_cnv2);
            std::cout << "export : " << filename << "\n" << std::endl;
            
            // write test image
            bb::WriteTestDataImage<float>("verilog/cifar10_test_160x120.ppm", 160, 120, td);
            bb::WriteTestDataImage<float>("verilog/cifar10_test_640x480.ppm", 640, 480, td);
        }
    }
}


// end of file
