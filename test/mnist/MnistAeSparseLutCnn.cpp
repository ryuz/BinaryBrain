// --------------------------------------------------------------------------
//  BinaryBrain  -- binary network evaluation platform
//   MNIST sample
//
//                                Copyright (C) 2018-2019 by Ryuji Fuchikami
// --------------------------------------------------------------------------


#include <iostream>

#include "bb/Sequential.h"
#include "bb/SparseLutN.h"
#include "bb/SparseLutDiscreteN.h"
#include "bb/BinaryLutN.h"
#include "bb/MaxPooling.h"
#include "bb/LoweringConvolution.h"
#include "bb/UpSampling.h"
#include "bb/BinaryModulation.h"
#include "bb/OptimizerAdam.h"
#include "bb/LossMeanSquaredError.h"
#include "bb/MetricsMeanSquaredError.h"
#include "bb/Runner.h"
#include "bb/LoadMnist.h"
#include "bb/ExportVerilog.h"


static void WritePgm(std::string fname, bb::FrameBuffer buf, int frame)
{
    std::ofstream ofs(fname);
    ofs << "P2\n";
    ofs << "28 28 \n";
    ofs << "255\n";
    for ( int i = 0; i < 28*28; ++i ) {
        ofs << (int)(buf.GetFP32(frame, i) * 255.0f) << "\n";
    }
}


template < typename T=float, class ModelType=bb::SparseLutN<6, T> >
void MnistAeSparseLutCnn_Tmp(int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read)
{
    std::string net_name = "MnistAeSparseLutCnn";

  // load MNIST data
#ifdef _DEBUG
    auto td = bb::LoadMnist<>::Load(10, 64, 32);
    std::cout << "!!! debug mode !!!" << std::endl;
#else
    auto td = bb::LoadMnist<>::Load();
#endif

    // 入力と出力を同じに
    td.t_shape = td.x_shape;
    td.t_train = td.x_train;
    td.t_test  = td.x_test;

    // create network
    auto enc_cnv0_sl0 = ModelType::Create(192);
    auto enc_cnv0_sl1 = ModelType::Create(32);
    auto enc_cnv1_sl0 = ModelType::Create(192);
    auto enc_cnv1_sl1 = ModelType::Create(32);
    auto enc_cnv2_sl0 = ModelType::Create(384);
    auto enc_cnv2_sl1 = ModelType::Create(64);
    auto enc_cnv3_sl0 = ModelType::Create(384);
    auto enc_cnv3_sl1 = ModelType::Create(64);
    auto enc_sl4      = ModelType::Create(1152);
    auto enc_sl5      = ModelType::Create(192);
    auto enc_sl6      = ModelType::Create(32);
    
    auto dec_sl6      = ModelType::Create(256);
    auto dec_sl5      = ModelType::Create(1024);
    auto dec_sl4      = ModelType::Create({7, 7, 64});
    auto dec_cnv3_sl0 = ModelType::Create(384);
    auto dec_cnv3_sl1 = ModelType::Create(64);
    auto dec_cnv2_sl0 = ModelType::Create(384);
    auto dec_cnv2_sl1 = ModelType::Create(64);
    auto dec_cnv1_sl0 = ModelType::Create(192);
    auto dec_cnv1_sl1 = ModelType::Create(32);
    auto dec_cnv0_sl0 = ModelType::Create(216);
    auto dec_cnv0_sl1 = ModelType::Create(36, false);
    auto dec_cnv0_sl2 = ModelType::Create(6, false);
    auto dec_cnv0_sl3 = ModelType::Create(1, false);


    {
        std::cout << "\n<Training>" << std::endl;

        // main network
        auto enc_cnv0_sub = bb::Sequential::Create();
        enc_cnv0_sub->Add(enc_cnv0_sl0);
        enc_cnv0_sub->Add(enc_cnv0_sl1);

        auto enc_cnv1_sub = bb::Sequential::Create();
        enc_cnv1_sub->Add(enc_cnv1_sl0);
        enc_cnv1_sub->Add(enc_cnv1_sl1);

        auto enc_cnv2_sub = bb::Sequential::Create();
        enc_cnv2_sub->Add(enc_cnv2_sl0);
        enc_cnv2_sub->Add(enc_cnv2_sl1);

        auto enc_cnv3_sub = bb::Sequential::Create();
        enc_cnv3_sub->Add(enc_cnv3_sl0);
        enc_cnv3_sub->Add(enc_cnv3_sl1);


        auto dec_cnv3_sub = bb::Sequential::Create();
        dec_cnv3_sub->Add(dec_cnv3_sl0);
        dec_cnv3_sub->Add(dec_cnv3_sl1);

        auto dec_cnv2_sub = bb::Sequential::Create();
        dec_cnv2_sub->Add(dec_cnv2_sl0);
        dec_cnv2_sub->Add(dec_cnv2_sl1);

        auto dec_cnv1_sub = bb::Sequential::Create();
        dec_cnv1_sub->Add(dec_cnv1_sl0);
        dec_cnv1_sub->Add(dec_cnv1_sl1);

        auto dec_cnv0_sub = bb::Sequential::Create();
        dec_cnv0_sub->Add(dec_cnv0_sl0);
        dec_cnv0_sub->Add(dec_cnv0_sl1);
        dec_cnv0_sub->Add(dec_cnv0_sl2);
        dec_cnv0_sub->Add(dec_cnv0_sl3);

        auto main_net = bb::Sequential::Create();
        main_net->Add(bb::LoweringConvolution<T>::Create(enc_cnv0_sub, 3, 3, 1, 1, "same"));    // 28x28
        main_net->Add(bb::LoweringConvolution<T>::Create(enc_cnv1_sub, 3, 3, 1, 1, "same"));    // 28x28
        main_net->Add(bb::MaxPooling<float>::Create(2, 2));
        main_net->Add(bb::LoweringConvolution<T>::Create(enc_cnv2_sub, 3, 3, 1, 1, "same"));    // 14x14
        main_net->Add(bb::LoweringConvolution<T>::Create(enc_cnv3_sub, 3, 3, 1, 1, "same"));    // 14x14
        main_net->Add(bb::MaxPooling<float>::Create(2, 2));
        main_net->Add(enc_sl4);
        main_net->Add(enc_sl5);
        main_net->Add(enc_sl6);

        main_net->Add(dec_sl6);
        main_net->Add(dec_sl5);
        main_net->Add(dec_sl4);
        main_net->Add(bb::UpSampling<T>::Create(2, 2));
        main_net->Add(bb::LoweringConvolution<T>::Create(dec_cnv3_sub, 3, 3, 1, 1, "same"));    // 14x14
        main_net->Add(bb::LoweringConvolution<T>::Create(dec_cnv2_sub, 3, 3, 1, 1, "same"));    // 14x14
        main_net->Add(bb::UpSampling<T>::Create(2, 2));
        main_net->Add(bb::LoweringConvolution<T>::Create(dec_cnv1_sub, 3, 3, 1, 1, "same"));    // 28x28
        main_net->Add(bb::LoweringConvolution<T>::Create(dec_cnv0_sub, 3, 3, 1, 1, "same"));    // 28x28

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
        runner_create.initial_evaluation = false; // file_read;       // ファイルを読んだ場合は最初に評価しておく
        auto runner = bb::Runner<float>::Create(runner_create);
        runner->Fitting(td, epoch_size, mini_batch_size);


        // write pgm
        bb::FrameBuffer x_buf(BB_TYPE_FP32, 8, {28, 28, 1});
        x_buf.SetVector(td.x_test, 0);
        auto y_buf = net->Forward(x_buf, false);
        WritePgm("out_0x.pgm", x_buf, 0);
        WritePgm("out_0y.pgm", y_buf, 0);
        WritePgm("out_1x.pgm", x_buf, 1);
        WritePgm("out_1y.pgm", y_buf, 1);
        WritePgm("out_2x.pgm", x_buf, 2);
        WritePgm("out_2y.pgm", y_buf, 2);
        WritePgm("out_3x.pgm", x_buf, 3);
        WritePgm("out_3y.pgm", y_buf, 3);
        WritePgm("out_4x.pgm", x_buf, 4);
        WritePgm("out_4y.pgm", y_buf, 4);
    }

#if 1
    {
        std::cout << "\n<Evaluation binary LUT-Network>" << std::endl;

        // LUT-network
        auto enc_cnv0_bl0 = bb::BinaryLutN<6, bb::Bit>::Create(enc_cnv0_sl0->GetOutputShape());
        auto enc_cnv0_bl1 = bb::BinaryLutN<6, bb::Bit>::Create(enc_cnv0_sl1->GetOutputShape());
        auto enc_cnv1_bl0 = bb::BinaryLutN<6, bb::Bit>::Create(enc_cnv1_sl0->GetOutputShape());
        auto enc_cnv1_bl1 = bb::BinaryLutN<6, bb::Bit>::Create(enc_cnv1_sl1->GetOutputShape());
        auto enc_cnv2_bl0 = bb::BinaryLutN<6, bb::Bit>::Create(enc_cnv2_sl0->GetOutputShape());
        auto enc_cnv2_bl1 = bb::BinaryLutN<6, bb::Bit>::Create(enc_cnv2_sl1->GetOutputShape());
        auto enc_cnv3_bl0 = bb::BinaryLutN<6, bb::Bit>::Create(enc_cnv3_sl0->GetOutputShape());
        auto enc_cnv3_bl1 = bb::BinaryLutN<6, bb::Bit>::Create(enc_cnv3_sl1->GetOutputShape());
        auto enc_bl4      = bb::BinaryLutN<6, bb::Bit>::Create(enc_sl4     ->GetOutputShape());
        auto enc_bl5      = bb::BinaryLutN<6, bb::Bit>::Create(enc_sl5     ->GetOutputShape());
        auto enc_bl6      = bb::BinaryLutN<6, bb::Bit>::Create(enc_sl6     ->GetOutputShape());
        
        auto dec_bl6      = bb::BinaryLutN<6, bb::Bit>::Create(dec_sl6     ->GetOutputShape());
        auto dec_bl5      = bb::BinaryLutN<6, bb::Bit>::Create(dec_sl5     ->GetOutputShape());
        auto dec_bl4      = bb::BinaryLutN<6, bb::Bit>::Create(dec_sl4     ->GetOutputShape());
        auto dec_cnv3_bl0 = bb::BinaryLutN<6, bb::Bit>::Create(dec_cnv3_sl0->GetOutputShape());
        auto dec_cnv3_bl1 = bb::BinaryLutN<6, bb::Bit>::Create(dec_cnv3_sl1->GetOutputShape());
        auto dec_cnv2_bl0 = bb::BinaryLutN<6, bb::Bit>::Create(dec_cnv2_sl0->GetOutputShape());
        auto dec_cnv2_bl1 = bb::BinaryLutN<6, bb::Bit>::Create(dec_cnv2_sl1->GetOutputShape());
        auto dec_cnv1_bl0 = bb::BinaryLutN<6, bb::Bit>::Create(dec_cnv1_sl0->GetOutputShape());
        auto dec_cnv1_bl1 = bb::BinaryLutN<6, bb::Bit>::Create(dec_cnv1_sl1->GetOutputShape());
        auto dec_cnv0_bl0 = bb::BinaryLutN<6, bb::Bit>::Create(dec_cnv0_sl0->GetOutputShape());
        auto dec_cnv0_bl1 = bb::BinaryLutN<6, bb::Bit>::Create(dec_cnv0_sl1->GetOutputShape());
        auto dec_cnv0_bl2 = bb::BinaryLutN<6, bb::Bit>::Create(dec_cnv0_sl2->GetOutputShape());
        auto dec_cnv0_bl3 = bb::BinaryLutN<6, bb::Bit>::Create(dec_cnv0_sl3->GetOutputShape());


        auto enc_cnv0_sub = bb::Sequential::Create();
        enc_cnv0_sub->Add(enc_cnv0_bl0);
        enc_cnv0_sub->Add(enc_cnv0_bl1);

        auto enc_cnv1_sub = bb::Sequential::Create();
        enc_cnv1_sub->Add(enc_cnv1_bl0);
        enc_cnv1_sub->Add(enc_cnv1_bl1);

        auto enc_cnv2_sub = bb::Sequential::Create();
        enc_cnv2_sub->Add(enc_cnv2_bl0);
        enc_cnv2_sub->Add(enc_cnv2_bl1);

        auto enc_cnv3_sub = bb::Sequential::Create();
        enc_cnv3_sub->Add(enc_cnv3_bl0);
        enc_cnv3_sub->Add(enc_cnv3_bl1);


        auto dec_cnv3_sub = bb::Sequential::Create();
        dec_cnv3_sub->Add(dec_cnv3_bl0);
        dec_cnv3_sub->Add(dec_cnv3_bl1);

        auto dec_cnv2_sub = bb::Sequential::Create();
        dec_cnv2_sub->Add(dec_cnv2_bl0);
        dec_cnv2_sub->Add(dec_cnv2_bl1);

        auto dec_cnv1_sub = bb::Sequential::Create();
        dec_cnv1_sub->Add(dec_cnv1_bl0);
        dec_cnv1_sub->Add(dec_cnv1_bl1);

        auto dec_cnv0_sub = bb::Sequential::Create();
        dec_cnv0_sub->Add(dec_cnv0_bl0);
        dec_cnv0_sub->Add(dec_cnv0_bl1);
        dec_cnv0_sub->Add(dec_cnv0_bl2);
        dec_cnv0_sub->Add(dec_cnv0_bl3);

        auto lut_net = bb::Sequential::Create();
        lut_net->Add(bb::LoweringConvolution<bb::Bit>::Create(enc_cnv0_sub, 3, 3, 1, 1, "same"));    // 28x28
        lut_net->Add(bb::LoweringConvolution<bb::Bit>::Create(enc_cnv1_sub, 3, 3, 1, 1, "same"));    // 28x28
        lut_net->Add(bb::MaxPooling<bb::Bit>::Create(2, 2));
        lut_net->Add(bb::LoweringConvolution<bb::Bit>::Create(enc_cnv2_sub, 3, 3, 1, 1, "same"));    // 14x14
        lut_net->Add(bb::LoweringConvolution<bb::Bit>::Create(enc_cnv3_sub, 3, 3, 1, 1, "same"));    // 14x14
        lut_net->Add(bb::MaxPooling<bb::Bit>::Create(2, 2));
        lut_net->Add(enc_bl4);
        lut_net->Add(enc_bl5);
        lut_net->Add(enc_bl6);

        lut_net->Add(dec_bl6);
        lut_net->Add(dec_bl5);
        lut_net->Add(dec_bl4);
        lut_net->Add(bb::UpSampling<bb::Bit>::Create(2, 2));
        lut_net->Add(bb::LoweringConvolution<bb::Bit>::Create(dec_cnv3_sub, 3, 3, 1, 1, "same"));    // 14x14
        lut_net->Add(bb::LoweringConvolution<bb::Bit>::Create(dec_cnv2_sub, 3, 3, 1, 1, "same"));    // 14x14
        lut_net->Add(bb::UpSampling<bb::Bit>::Create(2, 2));
        lut_net->Add(bb::LoweringConvolution<bb::Bit>::Create(dec_cnv1_sub, 3, 3, 1, 1, "same"));    // 28x28
        lut_net->Add(bb::LoweringConvolution<bb::Bit>::Create(dec_cnv0_sub, 3, 3, 1, 1, "same"));    // 28x28

        // evaluation network
        auto eval_net = bb::Sequential::Create();
        eval_net->Add(bb::BinaryModulation<bb::Bit>::Create(lut_net, test_modulation_size));

        // set input shape
        eval_net->SetInputShape(td.x_shape);


        // テーブル化して取り込み(現状まだSetInputShape後の取り込みが必要)
        std::cout << "parameter copy to binary LUT-Network" << std::endl;
        enc_cnv0_bl0->ImportLayer(enc_cnv0_sl0);
        enc_cnv0_bl1->ImportLayer(enc_cnv0_sl1);
        enc_cnv1_bl0->ImportLayer(enc_cnv1_sl0);
        enc_cnv1_bl1->ImportLayer(enc_cnv1_sl1);
        enc_cnv2_bl0->ImportLayer(enc_cnv2_sl0);
        enc_cnv2_bl1->ImportLayer(enc_cnv2_sl1);
        enc_cnv3_bl0->ImportLayer(enc_cnv3_sl0);
        enc_cnv3_bl1->ImportLayer(enc_cnv3_sl1);
        enc_bl4     ->ImportLayer(enc_sl4     );
        enc_bl5     ->ImportLayer(enc_sl5     );
        enc_bl6     ->ImportLayer(enc_sl6     );

        dec_bl6     ->ImportLayer(dec_sl6     );
        dec_bl5     ->ImportLayer(dec_sl5     );
        dec_bl4     ->ImportLayer(dec_sl4     );
        dec_cnv3_bl0->ImportLayer(dec_cnv3_sl0);
        dec_cnv3_bl1->ImportLayer(dec_cnv3_sl1);
        dec_cnv2_bl0->ImportLayer(dec_cnv2_sl0);
        dec_cnv2_bl1->ImportLayer(dec_cnv2_sl1);
        dec_cnv1_bl0->ImportLayer(dec_cnv1_sl0);
        dec_cnv1_bl1->ImportLayer(dec_cnv1_sl1);
        dec_cnv0_bl0->ImportLayer(dec_cnv0_sl0);
        dec_cnv0_bl1->ImportLayer(dec_cnv0_sl1);
        dec_cnv0_bl2->ImportLayer(dec_cnv0_sl2);
        dec_cnv0_bl3->ImportLayer(dec_cnv0_sl3);

        // 評価
        if ( 1 ) {
            std::cout << "test_modulation_size  : " << test_modulation_size  << std::endl;
            bb::Runner<float>::create_t lut_runner_create;
            lut_runner_create.name        = "Lut_" + net_name;
            lut_runner_create.net         = eval_net;
            lut_runner_create.lossFunc    = bb::LossMeanSquaredError<float>::Create();
            lut_runner_create.metricsFunc = bb::MetricsMeanSquaredError<float>::Create();
            lut_runner_create.optimizer   = bb::OptimizerAdam<float>::Create();
            lut_runner_create.initial_evaluation = false;
            lut_runner_create.print_progress = true;    // 途中結果を出力
            auto lut_runner = bb::Runner<float>::Create(lut_runner_create);
            auto lut_accuracy = lut_runner->Evaluation(td, mini_batch_size);
            std::cout << "lut_accuracy : " << lut_accuracy << std::endl;

            // write pgm
            bb::FrameBuffer x_buf(BB_TYPE_FP32, 32, {28, 28, 1});
            x_buf.SetVector(td.x_test, 0);
            auto y_buf = eval_net->Forward(x_buf, false);
            WritePgm("lut_0x.pgm", x_buf, 0);
            WritePgm("lut_0y.pgm", y_buf, 0);
            WritePgm("lut_1x.pgm", x_buf, 1);
            WritePgm("lut_1y.pgm", y_buf, 1);
            WritePgm("lut_2x.pgm", x_buf, 2);
            WritePgm("lut_2y.pgm", y_buf, 2);
            WritePgm("lut_3x.pgm", x_buf, 3);
            WritePgm("lut_3y.pgm", y_buf, 3);
            WritePgm("lut_4x.pgm", x_buf, 4);
            WritePgm("lut_4y.pgm", y_buf, 4);
        }

#if 0
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
#endif
    }
#endif
}


void MnistAeSparseLutCnn(int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read)
{
#ifdef BB_WITH_CUDA
    if ( binary_mode ) {
        MnistAeSparseLutCnn_Tmp< float, bb::SparseLutN<6, float> >(epoch_size, mini_batch_size, train_modulation_size, test_modulation_size, binary_mode, file_read);
    }
    else {
        MnistAeSparseLutCnn_Tmp< bb::Bit, bb::SparseLutN<6, bb::Bit> >(epoch_size, mini_batch_size, train_modulation_size, test_modulation_size, binary_mode, file_read);
    }
#else
    MnistAeSparseLutCnn_Tmp< float, bb::SparseLutDiscreteN<6, float> >(epoch_size, mini_batch_size, train_modulation_size, test_modulation_size, binary_mode, file_read);
#endif
}



// end of file
