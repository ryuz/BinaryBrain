// --------------------------------------------------------------------------
//  BinaryBrain  -- binary network evaluation platform
//   CIFAR-10 sample
//
//                                Copyright (C) 2018-2019 by Ryuji Fuchikami
// --------------------------------------------------------------------------

#include <iostream>

#include "bb/Sequential.h"
#include "bb/BinaryModulation.h"
#include "bb/Reduce.h"
#include "bb/MicroMlp.h"
#include "bb/BinaryLutN.h"
#include "bb/Convolution2d.h"
#include "bb/MaxPooling.h"
#include "bb/OptimizerAdam.h"
#include "bb/LossSoftmaxCrossEntropy.h"
#include "bb/MetricsCategoricalAccuracy.h"
#include "bb/Runner.h"
#include "bb/LoadCifar10.h"
#include "bb/ExportVerilog.h"


void Cifar10MicroMlpLutCnn(int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read)
{
    std::string net_name       = "Cifar10MicroMlpLutCnn";
    std::string velilog_path   = "../../verilog/cifar10/";
    std::string velilog_module = "Cifar10LutCnn";

  // load cifar-10 data
#ifdef _DEBUG
    auto td = bb::LoadCifar10<>::Load(1);
    std::cout << "!!! debug mode !!!" << std::endl;
#else
    auto td = bb::LoadCifar10<>::Load();
#endif

    // create network
    auto layer_cnv0_mm0 = bb::MicroMlp<>::Create(256);
    auto layer_cnv0_mm1 = bb::MicroMlp<>::Create(192);
    auto layer_cnv0_mm2 = bb::MicroMlp<>::Create(32);
    auto layer_cnv1_mm0 = bb::MicroMlp<>::Create(256);
    auto layer_cnv1_mm1 = bb::MicroMlp<>::Create(192);
    auto layer_cnv1_mm2 = bb::MicroMlp<>::Create(32);
    auto layer_cnv2_mm0 = bb::MicroMlp<>::Create(512);
    auto layer_cnv2_mm1 = bb::MicroMlp<>::Create(384);
    auto layer_cnv2_mm2 = bb::MicroMlp<>::Create(64);
    auto layer_cnv3_mm0 = bb::MicroMlp<>::Create(512);
    auto layer_cnv3_mm1 = bb::MicroMlp<>::Create(384);
    auto layer_cnv3_mm2 = bb::MicroMlp<>::Create(64);
    auto layer_mm4      = bb::MicroMlp<>::Create(1024);
    auto layer_mm5      = bb::MicroMlp<>::Create(420);
    auto layer_mm6      = bb::MicroMlp<>::Create(70);

    {
        std::cout << "\n<Training>" << std::endl;

        auto cnv0_sub = bb::Sequential::Create();
        cnv0_sub->Add(layer_cnv0_mm0);
        cnv0_sub->Add(layer_cnv0_mm1);
        cnv0_sub->Add(layer_cnv0_mm2);

        auto cnv1_sub = bb::Sequential::Create();
        cnv1_sub->Add(layer_cnv1_mm0);
        cnv1_sub->Add(layer_cnv1_mm1);
        cnv1_sub->Add(layer_cnv1_mm2);

        auto cnv2_sub = bb::Sequential::Create();
        cnv2_sub->Add(layer_cnv2_mm0);
        cnv2_sub->Add(layer_cnv2_mm1);
        cnv2_sub->Add(layer_cnv2_mm2);

        auto cnv3_sub = bb::Sequential::Create();
        cnv3_sub->Add(layer_cnv3_mm0);
        cnv3_sub->Add(layer_cnv3_mm1);
        cnv3_sub->Add(layer_cnv3_mm2);
        
        auto main_net = bb::Sequential::Create();
        main_net->Add(bb::Convolution2d<>::Create(cnv0_sub, 3, 3));
        main_net->Add(bb::Convolution2d<>::Create(cnv1_sub, 3, 3));
        main_net->Add(bb::MaxPooling<>::Create(2, 2));
        main_net->Add(bb::Convolution2d<>::Create(cnv2_sub, 3, 3));
        main_net->Add(bb::Convolution2d<>::Create(cnv3_sub, 3, 3));
        main_net->Add(bb::MaxPooling<>::Create(2, 2));
        main_net->Add(layer_mm4);
        main_net->Add(layer_mm5);
        main_net->Add(layer_mm6);
        
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
    }


    {
        std::cout << "\n<Evaluation binary LUT-Network>" << std::endl;

        // LUT-network
        auto layer_cnv0_bl0 = bb::BinaryLutN<>::Create(layer_cnv0_mm0->GetOutputShape());
        auto layer_cnv0_bl1 = bb::BinaryLutN<>::Create(layer_cnv0_mm1->GetOutputShape());
        auto layer_cnv0_bl2 = bb::BinaryLutN<>::Create(layer_cnv0_mm2->GetOutputShape());
        auto layer_cnv1_bl0 = bb::BinaryLutN<>::Create(layer_cnv1_mm0->GetOutputShape());
        auto layer_cnv1_bl1 = bb::BinaryLutN<>::Create(layer_cnv1_mm1->GetOutputShape());
        auto layer_cnv1_bl2 = bb::BinaryLutN<>::Create(layer_cnv1_mm2->GetOutputShape());
        auto layer_cnv2_bl0 = bb::BinaryLutN<>::Create(layer_cnv2_mm0->GetOutputShape());
        auto layer_cnv2_bl1 = bb::BinaryLutN<>::Create(layer_cnv2_mm1->GetOutputShape());
        auto layer_cnv2_bl2 = bb::BinaryLutN<>::Create(layer_cnv2_mm2->GetOutputShape());
        auto layer_cnv3_bl0 = bb::BinaryLutN<>::Create(layer_cnv3_mm0->GetOutputShape());
        auto layer_cnv3_bl1 = bb::BinaryLutN<>::Create(layer_cnv3_mm1->GetOutputShape());
        auto layer_cnv3_bl2 = bb::BinaryLutN<>::Create(layer_cnv3_mm2->GetOutputShape());
        auto layer_bl4      = bb::BinaryLutN<>::Create(layer_mm4->GetOutputShape());
        auto layer_bl5      = bb::BinaryLutN<>::Create(layer_mm5->GetOutputShape());
        auto layer_bl6      = bb::BinaryLutN<>::Create(layer_mm6->GetOutputShape());

        auto cnv0_sub = bb::Sequential::Create();
        cnv0_sub->Add(layer_cnv0_bl0);
        cnv0_sub->Add(layer_cnv0_bl1);
        cnv0_sub->Add(layer_cnv0_bl2);

        auto cnv1_sub = bb::Sequential::Create();
        cnv1_sub->Add(layer_cnv1_bl0);
        cnv1_sub->Add(layer_cnv1_bl1);
        cnv1_sub->Add(layer_cnv1_bl2);

        auto cnv2_sub = bb::Sequential::Create();
        cnv2_sub->Add(layer_cnv2_bl0);
        cnv2_sub->Add(layer_cnv2_bl1);
        cnv2_sub->Add(layer_cnv2_bl2);

        auto cnv3_sub = bb::Sequential::Create();
        cnv3_sub->Add(layer_cnv3_bl0);
        cnv3_sub->Add(layer_cnv3_bl1);
        cnv3_sub->Add(layer_cnv3_bl2);

        auto cnv4_sub = bb::Sequential::Create();
        cnv4_sub->Add(layer_bl4);
        cnv4_sub->Add(layer_bl5);
        cnv4_sub->Add(layer_bl6);

        auto cnv0 = bb::Convolution2d<bb::Bit>::Create(cnv0_sub, 3, 3);
        auto cnv1 = bb::Convolution2d<bb::Bit>::Create(cnv1_sub, 3, 3);
        auto pol0 = bb::MaxPooling<bb::Bit>::Create(2, 2);

        auto cnv2 = bb::Convolution2d<bb::Bit>::Create(cnv2_sub, 3, 3);
        auto cnv3 = bb::Convolution2d<bb::Bit>::Create(cnv3_sub, 3, 3);
        auto pol1 = bb::MaxPooling<bb::Bit>::Create(2, 2);

        // 32x32 以外も入力できるように最終段も畳み込みに変換
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
        auto eval_net = bb::Sequential::Create();
        eval_net->Add(bb::BinaryModulation<bb::Bit>::Create(lut_net, test_modulation_size));
        eval_net->Add(bb::Reduce<>::Create(td.t_shape));

        // set input shape
        eval_net->SetInputShape(td.x_shape);
        
        // テーブル化して取り込み(現状まだSetInputShape後の取り込みが必要)
        std::cout << "parameter copy to LUT-Network" << std::endl;
        layer_cnv0_bl0->ImportLayer(layer_cnv0_mm0);
        layer_cnv0_bl1->ImportLayer(layer_cnv0_mm1);
        layer_cnv0_bl2->ImportLayer(layer_cnv0_mm2);
        layer_cnv1_bl0->ImportLayer(layer_cnv1_mm0);
        layer_cnv1_bl1->ImportLayer(layer_cnv1_mm1);
        layer_cnv1_bl2->ImportLayer(layer_cnv1_mm2);
        layer_cnv2_bl0->ImportLayer(layer_cnv2_mm0);
        layer_cnv2_bl1->ImportLayer(layer_cnv2_mm1);
        layer_cnv2_bl2->ImportLayer(layer_cnv2_mm2);
        layer_cnv3_bl0->ImportLayer(layer_cnv3_mm0);
        layer_cnv3_bl1->ImportLayer(layer_cnv3_mm1);
        layer_cnv3_bl2->ImportLayer(layer_cnv3_mm2);
        layer_bl4     ->ImportLayer(layer_mm4);
        layer_bl5     ->ImportLayer(layer_mm5);
        layer_bl6     ->ImportLayer(layer_mm6);

        // 評価
        if ( 1 ) {
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

            std::string filename = velilog_path + velilog_module + ".v";
            std::ofstream ofs(filename);
            ofs << "`timescale 1ns / 1ps\n\n";
            bb::ExportVerilog_LutCnnLayersAxi4s(ofs, velilog_module + "Cnv0", vec_cnv0);
            bb::ExportVerilog_LutCnnLayersAxi4s(ofs, velilog_module + "Cnv1", vec_cnv1);
            bb::ExportVerilog_LutCnnLayersAxi4s(ofs, velilog_module + "Cnv2", vec_cnv2);
            std::cout << "export : " << filename << "\n" << std::endl;
            
            // write test image
//          bb::WriteTestDataImage<float>(velilog_path + "cifar10_test_160x120.ppm", 160, 120, td);
//          bb::WriteTestDataImage<float>(velilog_path + "cifar10_test_640x480.ppm", 640, 480, td);
        }
    }
}


// end of file
