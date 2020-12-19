// --------------------------------------------------------------------------
//  BinaryBrain  -- binary network evaluation platform
//   CIFAR-10 sample
//
//                                Copyright (C) 2018-2019 by Ryuji Fuchikami
// --------------------------------------------------------------------------


#include <iostream>

#include "bb/Sequential.h"
#include "bb/DifferentiableLutN.h"
#include "bb/DifferentiableLutDiscreteN.h"
#include "bb/BinaryLutN.h"
#include "bb/Reduce.h"
#include "bb/BinaryModulation.h"
#include "bb/OptimizerAdam.h"
//#include "bb/LossSoftmaxCrossEntropy.h"
//#include "bb/MetricsCategoricalAccuracy.h"
#include "bb/LossMeanSquaredError.h"
#include "bb/MetricsMeanSquaredError.h"
#include "bb/Runner.h"
#include "bb/LoadCifar10.h"
#include "bb/ExportVerilog.h"
#include "bb/NormalDistributionGenerator.h"
#include "bb/UniformDistributionGenerator.h"
#include "bb/PnmImage.h"

#include "bb/DenseAffine.h"
#include "bb/BatchNormalization.h"


template<typename T=bb::Bit>
void NrDifferentiableLutCnn(int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read)
{
    std::string net_name = "Cifar10NrDifferentiableLutCnn";

    if ( bb::DataType<T>::type == BB_TYPE_BIT ) {
        binary_mode = true;
    }

  // load cifar-10 data
#ifdef _DEBUG
    auto td = bb::LoadCifar10<>::Load(1);
    std::cout << "!!! debug mode !!!" << std::endl;
#else
    auto td = bb::LoadCifar10<>::Load();
#endif

#if 1
    // 入力をバックアップ
    auto org_train = td.x_train;
    auto org_test  = td.x_test;

     // 入力と出力を同じに
    td.t_shape = td.x_shape;
    td.t_train = td.x_train;
    td.t_test  = td.x_test;

//  auto  noise_gen = bb::NormalDistributionGenerator<float>::Create(0.0f, 0.1f);
    auto  noise_gen = bb::UniformDistributionGenerator<float>::Create(0.0f, 1.0f);

    // 入力にノイズ付与
    for ( auto& x : td.x_train ) {
        for ( auto& v : x ) {
//            v += noise_gen->GetValue();
//            v = 0.5f + noise_gen->GetValue();
            v = noise_gen->GetValue();
            v = std::max(v, 0.0f);
            v = std::min(v, 1.0f);
        }
    }
    for ( auto& x : td.x_test ) {
        for ( auto& v : x ) {
//          v += noise_gen->GetValue();
//            v = 0.5f + noise_gen->GetValue();
            v = noise_gen->GetValue();
            v = std::max(v, 0.0f);
            v = std::min(v, 1.0f);
        }
    }

    // 差分を期待値に
    for ( size_t i = 0; i < td.t_train.size(); ++i ) {
        for ( size_t j = 0; j < td.t_train[i].size(); ++j ) {
            td.t_train[i][j] = 0.5f;// + td.x_train[i][j] - org_train[i][j];
        }
    }
    for ( size_t i = 0; i < td.t_test.size(); ++i ) {
        for ( size_t j = 0; j < td.t_test[i].size(); ++j ) {
            td.t_test[i][j] = 0.5f;// + td.x_test[i][j] - org_test[i][j];
        }
    }

#else
    // 入力と出力を同じに
    td.t_shape = td.x_shape;
    td.t_train = td.x_train;
    td.t_test  = td.x_test;

    auto  noise_gen = bb::NormalDistributionGenerator<float>::Create(0.0f, 0.1f);

    // ノイズ付与
    for ( auto& x : td.x_train ) {
        for ( auto& v : x ) {
            v += noise_gen->GetValue();
            v = std::max(v, 0.0f);
            v = std::min(v, 1.0f);
        }
    }
    for ( auto& x : td.x_test ) {
        for ( auto& v : x ) {
            v += noise_gen->GetValue();
            v = std::max(v, 0.0f);
            v = std::min(v, 1.0f);
        }
    }
#endif

    {
        // write pgm
        bb::FrameBuffer x_buf(8, {32, 32, 3}, BB_TYPE_FP32);
        x_buf.SetVector(td.x_test, 0);
//      auto y_buf = net->Forward(x_buf, false);
        bb::WritePpm("td0_x_test.ppm", x_buf, 32, 32, 0);
        bb::WritePpm("td1_x_test.ppm", x_buf, 32, 32, 1);
        bb::WritePpm("td2_x_test.ppm", x_buf, 32, 32, 2);
        bb::WritePpm("td3_x_test.ppm", x_buf, 32, 32, 3);
        bb::WritePpm("td4_x_test.ppm", x_buf, 32, 32, 4);

        x_buf.SetVector(td.t_test, 0);
        bb::WritePpm("td0_t_test.ppm", x_buf, 32, 32, 0);
        bb::WritePpm("td1_t_test.ppm", x_buf, 32, 32, 1);
        bb::WritePpm("td2_t_test.ppm", x_buf, 32, 32, 2);
        bb::WritePpm("td3_t_test.ppm", x_buf, 32, 32, 3);
        bb::WritePpm("td4_t_test.ppm", x_buf, 32, 32, 4);

        x_buf.SetVector(org_test, 0);
        bb::WritePpm("td0_o_test.ppm", x_buf, 32, 32, 0);
        bb::WritePpm("td1_o_test.ppm", x_buf, 32, 32, 1);
        bb::WritePpm("td2_o_test.ppm", x_buf, 32, 32, 2);
        bb::WritePpm("td3_o_test.ppm", x_buf, 32, 32, 3);
        bb::WritePpm("td4_o_test.ppm", x_buf, 32, 32, 4);
    }


    // create network
    auto layer_cnv0_sl0 = bb::DifferentiableLutN<6, T>::Create(192,  true);
    auto layer_cnv0_sl1 = bb::DifferentiableLutN<6, T>::Create(32,   false);

//  auto layer_cnv1_sl0 = bb::DifferentiableLutN<6, T>::Create(1152, true);
//  auto layer_cnv1_sl1 = bb::DifferentiableLutN<6, T>::Create(192,  true);
//  auto layer_cnv1_sl2 = bb::DifferentiableLutN<6, T>::Create(32,   false);

//  auto layer_cnv2_sl0 = bb::DifferentiableLutN<6, T>::Create(1152, true);
//  auto layer_cnv2_sl1 = bb::DifferentiableLutN<6, T>::Create(192,  true);
//  auto layer_cnv2_sl2 = bb::DifferentiableLutN<6, T>::Create(32,   false);

//  auto layer_cnv3_sl0 = bb::DifferentiableLutN<6, T>::Create(1152, true);
    auto layer_cnv3_sl1 = bb::DifferentiableLutN<6, T>::Create(192,  true);
    auto layer_cnv3_sl2 = bb::DifferentiableLutN<6, T>::Create(32,   false);
    auto layer_cnv3_sl3 = bb::DifferentiableLutN<6, T>::Create(3,    false);

    {
        std::cout << "\n<Training>" << std::endl;

        auto cnv0_sub = bb::Sequential::Create();
        cnv0_sub->Add(layer_cnv0_sl0);
        cnv0_sub->Add(layer_cnv0_sl1);

//        auto cnv1_sub = bb::Sequential::Create();
//        cnv1_sub->Add(layer_cnv1_sl0);
//        cnv1_sub->Add(layer_cnv1_sl1);
//        cnv1_sub->Add(layer_cnv1_sl2);

//        auto cnv2_sub = bb::Sequential::Create();
//        cnv2_sub->Add(layer_cnv2_sl0);
//        cnv2_sub->Add(layer_cnv2_sl1);
//        cnv2_sub->Add(layer_cnv2_sl2);

        auto cnv3_sub = bb::Sequential::Create();
//        cnv3_sub->Add(layer_cnv3_sl0);
        cnv3_sub->Add(layer_cnv3_sl1);
        cnv3_sub->Add(layer_cnv3_sl2);
        cnv3_sub->Add(layer_cnv3_sl3);


        auto main_net = bb::Sequential::Create();
        main_net->Add(bb::LoweringConvolution<T>::Create(cnv0_sub, 3, 3, 1, 1, "same"));
//        main_net->Add(bb::LoweringConvolution<T>::Create(cnv1_sub, 3, 3, 1, 1, "same"));
//        main_net->Add(bb::LoweringConvolution<T>::Create(cnv2_sub, 3, 3, 1, 1, "same"));
        main_net->Add(bb::LoweringConvolution<T>::Create(cnv3_sub, 3, 3, 1, 1, "same"));

        // modulation wrapper
        auto net = bb::Sequential::Create();
        net->Add(bb::BinaryModulation<T>::Create(main_net, train_modulation_size, test_modulation_size));

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
        runner_create.initial_evaluation = false;// file_read;       // ファイルを読んだ場合は最初に評価しておく
        auto runner = bb::Runner<float>::Create(runner_create);
        runner->Fitting(td, epoch_size, mini_batch_size);

        // write pgm
        bb::FrameBuffer x_buf( 8, {32, 32, 3}, BB_TYPE_FP32);
        x_buf.SetVector(td.x_test, 0);
        auto y_buf = net->Forward(x_buf, false);
        bb::WritePpm("out_0x.ppm", x_buf, 32, 32, 0);
        bb::WritePpm("out_0y.ppm", y_buf, 32, 32, 0);
        bb::WritePpm("out_1x.ppm", x_buf, 32, 32, 1);
        bb::WritePpm("out_1y.ppm", y_buf, 32, 32, 1);
        bb::WritePpm("out_2x.ppm", x_buf, 32, 32, 2);
        bb::WritePpm("out_2y.ppm", y_buf, 32, 32, 2);
        bb::WritePpm("out_3x.ppm", x_buf, 32, 32, 3);
        bb::WritePpm("out_3y.ppm", y_buf, 32, 32, 3);
        bb::WritePpm("out_4x.ppm", x_buf, 32, 32, 4);
        bb::WritePpm("out_4y.ppm", y_buf, 32, 32, 4);

        auto z_buf = x_buf - y_buf + 0.5f;
        bb::WritePpm("out_0z.ppm", z_buf, 32, 32, 0);
        bb::WritePpm("out_1z.ppm", z_buf, 32, 32, 1);
        bb::WritePpm("out_2z.ppm", z_buf, 32, 32, 2);
        bb::WritePpm("out_3z.ppm", z_buf, 32, 32, 3);
        bb::WritePpm("out_4z.ppm", z_buf, 32, 32, 4);

        bb::FrameBuffer t_buf(8, {32, 32, 3}, BB_TYPE_FP32);
        t_buf.SetVector(td.t_test, 0);

        z_buf = x_buf - t_buf + 0.5f;
        bb::WritePpm("out_0zt.ppm", z_buf, 32, 32, 0);
        bb::WritePpm("out_1zt.ppm", z_buf, 32, 32, 1);
        bb::WritePpm("out_2zt.ppm", z_buf, 32, 32, 2);
        bb::WritePpm("out_3zt.ppm", z_buf, 32, 32, 3);
        bb::WritePpm("out_4zt.ppm", z_buf, 32, 32, 4);
    }

#if 0
    {
        std::cout << "\n<Evaluation binary LUT-Network>" << std::endl;

        // LUT-network
        auto layer_cnv0_bl0 = bb::BinaryLutN<6, bb::Bit>::Create(layer_cnv0_sl0->GetOutputShape());
        auto layer_cnv0_bl1 = bb::BinaryLutN<6, bb::Bit>::Create(layer_cnv0_sl1->GetOutputShape());
        auto layer_cnv1_bl0 = bb::BinaryLutN<6, bb::Bit>::Create(layer_cnv1_sl0->GetOutputShape());
        auto layer_cnv1_bl1 = bb::BinaryLutN<6, bb::Bit>::Create(layer_cnv1_sl1->GetOutputShape());
        auto layer_cnv1_bl2 = bb::BinaryLutN<6, bb::Bit>::Create(layer_cnv1_sl2->GetOutputShape());
        auto layer_cnv2_bl0 = bb::BinaryLutN<6, bb::Bit>::Create(layer_cnv2_sl0->GetOutputShape());
        auto layer_cnv2_bl1 = bb::BinaryLutN<6, bb::Bit>::Create(layer_cnv2_sl1->GetOutputShape());
        auto layer_cnv2_bl2 = bb::BinaryLutN<6, bb::Bit>::Create(layer_cnv2_sl2->GetOutputShape());
        auto layer_cnv3_bl0 = bb::BinaryLutN<6, bb::Bit>::Create(layer_cnv3_sl0->GetOutputShape());
        auto layer_cnv3_bl1 = bb::BinaryLutN<6, bb::Bit>::Create(layer_cnv3_sl1->GetOutputShape());
        auto layer_cnv3_bl2 = bb::BinaryLutN<6, bb::Bit>::Create(layer_cnv3_sl2->GetOutputShape());
        auto layer_bl4      = bb::BinaryLutN<6, bb::Bit>::Create(layer_sl4->GetOutputShape());
        auto layer_bl5      = bb::BinaryLutN<6, bb::Bit>::Create(layer_sl5->GetOutputShape());
        auto layer_bl6      = bb::BinaryLutN<6, bb::Bit>::Create(layer_sl6->GetOutputShape());
        auto layer_bl7      = bb::BinaryLutN<6, bb::Bit>::Create(layer_sl7->GetOutputShape());
        auto layer_bl8      = bb::BinaryLutN<6, bb::Bit>::Create(layer_sl8->GetOutputShape());
        auto layer_bl9      = bb::BinaryLutN<6, bb::Bit>::Create(layer_sl9->GetOutputShape());
        auto layer_bl10     = bb::BinaryLutN<6, bb::Bit>::Create(layer_sl10->GetOutputShape());

        auto cnv0_sub = bb::Sequential::Create();
        cnv0_sub->Add(layer_cnv0_bl0);
        cnv0_sub->Add(layer_cnv0_bl1);

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
        cnv4_sub->Add(layer_bl7);
        cnv4_sub->Add(layer_bl8);
        cnv4_sub->Add(layer_bl9);
        cnv4_sub->Add(layer_bl10);

        auto cnv0 = bb::LoweringConvolution<bb::Bit>::Create(cnv0_sub, 3, 3);
        auto cnv1 = bb::LoweringConvolution<bb::Bit>::Create(cnv1_sub, 3, 3);
        auto pol0 = bb::MaxPooling<bb::Bit>::Create(2, 2);

        auto cnv2 = bb::LoweringConvolution<bb::Bit>::Create(cnv2_sub, 3, 3);
        auto cnv3 = bb::LoweringConvolution<bb::Bit>::Create(cnv3_sub, 3, 3);
        auto pol1 = bb::MaxPooling<bb::Bit>::Create(2, 2);

        auto cnv4 = bb::LoweringConvolution<bb::Bit>::Create(cnv4_sub, 4, 4);

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
        std::cout << "parameter copy to binary LUT-Network" << std::endl;
        layer_cnv0_bl0->ImportLayer(layer_cnv0_sl0);
        layer_cnv0_bl1->ImportLayer(layer_cnv0_sl1);
        layer_cnv1_bl2->ImportLayer(layer_cnv1_sl2);
        layer_cnv1_bl1->ImportLayer(layer_cnv1_sl1);
        layer_cnv1_bl2->ImportLayer(layer_cnv1_sl2);
        layer_cnv2_bl0->ImportLayer(layer_cnv2_sl0);
        layer_cnv2_bl1->ImportLayer(layer_cnv2_sl1);
        layer_cnv2_bl2->ImportLayer(layer_cnv2_sl2);
        layer_cnv3_bl0->ImportLayer(layer_cnv3_sl0);
        layer_cnv3_bl1->ImportLayer(layer_cnv3_sl1);
        layer_cnv3_bl2->ImportLayer(layer_cnv3_sl2);
        layer_bl4     ->ImportLayer(layer_sl4);
        layer_bl5     ->ImportLayer(layer_sl5);
        layer_bl6     ->ImportLayer(layer_sl6);
        layer_bl7     ->ImportLayer(layer_sl7);
        layer_bl8     ->ImportLayer(layer_sl8);
        layer_bl9     ->ImportLayer(layer_sl9);
        layer_bl10    ->ImportLayer(layer_sl10);

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
            std::vector< std::shared_ptr< bb::Filter2d<bb::Bit> > >  vec_cnv0;
            std::vector< std::shared_ptr< bb::Filter2d<bb::Bit> > >  vec_cnv1;
            std::vector< std::shared_ptr< bb::Filter2d<bb::Bit> > >  vec_cnv2;

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
            bb::WriteTestDataImage<float>("verilog/mnist_test_160x120.ppm", 160, 120, td);
            bb::WriteTestDataImage<float>("verilog/mnist_test_640x480.ppm", 640, 480, td);
        }
    }
#endif
}


void Cifar10NrDifferentiableLutCnn(int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read)
{
    if ( binary_mode ) {
        NrDifferentiableLutCnn<bb::Bit>(epoch_size, mini_batch_size, train_modulation_size, test_modulation_size, binary_mode, file_read);
    }
    else {
//      NrDifferentiableLutCnn<float>(epoch_size, mini_batch_size, train_modulation_size, test_modulation_size, binary_mode, file_read);
    }
}


// end of file
