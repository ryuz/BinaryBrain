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

#include "bb/StochasticLut6.h"
#include "bb/BinaryLutN.h"
#include "bb/MetricsMeanSquaredError.h"
#include "bb/LossMeanSquaredError.h"
#include "bb/OptimizerAdam.h"
#include "bb/OptimizerSgd.h"
#include "bb/Sequential.h"
#include "bb/Runner.h"
#include "bb/RealToBinary.h"
#include "bb/BinaryToReal.h"
#include "bb/UniformDistributionGenerator.h"
#include "bb/ExportVerilog.h"


#include "LoadDiabetes.h"



void DiabetesRegressionStochasticLut6(int epoch_size, size_t mini_batch_size)
{
    // load diabetes data
    auto td = LoadDiabetes<>();

    bb::TrainDataNormalize(td);

    auto layer_sl0 = bb::StochasticLut6<>::Create({ 1024 });
    auto layer_sl1 = bb::StochasticLut6<>::Create({ 512 });
    auto layer_sl2 = bb::StochasticLut6<>::Create({ 216 });
    auto layer_sl3 = bb::StochasticLut6<>::Create({ 36 });
    auto layer_sl4 = bb::StochasticLut6<>::Create({ 6 });
    auto layer_sl5 = bb::StochasticLut6<>::Create({ 1 });

    {
        // 確率的LUTで学習
        auto net = bb::Sequential::Create();
        net->Add(layer_sl0);
        net->Add(layer_sl1);
        net->Add(layer_sl2);
        net->Add(layer_sl3);
        net->Add(layer_sl4);
        net->Add(layer_sl5);
        net->SetInputShape({ 10 });

        bb::Runner<float>::create_t runner_create;
        runner_create.name        = "DiabetesRegressionStochasticLut6";
        runner_create.net         = net;
        runner_create.lossFunc    = bb::LossMeanSquaredError<float>::Create();
        runner_create.metricsFunc = bb::MetricsMeanSquaredError<float>::Create();
    //  runner_create.optimizer = bb::OptimizerSgd<float>::Create(0.00001f);
        runner_create.optimizer = bb::OptimizerAdam<float>::Create();
        runner_create.file_read = false;
        runner_create.file_write = true;
        runner_create.write_serial = false;
        runner_create.print_progress = false;
        runner_create.initial_evaluation = true;
        auto runner = bb::Runner<float>::Create(runner_create);

        runner->Fitting(td, epoch_size, mini_batch_size);
    }

    {
        // LUT-network
        int mux_size = 255;

        auto layer_lut0 = bb::BinaryLutN<>::Create(layer_sl0->GetOutputShape());
        auto layer_lut1 = bb::BinaryLutN<>::Create(layer_sl1->GetOutputShape());
        auto layer_lut2 = bb::BinaryLutN<>::Create(layer_sl2->GetOutputShape());
        auto layer_lut3 = bb::BinaryLutN<>::Create(layer_sl3->GetOutputShape());
        auto layer_lut4 = bb::BinaryLutN<>::Create(layer_sl4->GetOutputShape());
        auto layer_lut5 = bb::BinaryLutN<>::Create(layer_sl5->GetOutputShape());

        auto lut_net = bb::Sequential::Create();
        lut_net->Add(bb::RealToBinary<float, bb::Bit>::Create(mux_size, bb::UniformDistributionGenerator<float>::Create(0.0f, 1.0f, 1)));
        lut_net->Add(layer_lut0);
        lut_net->Add(layer_lut1);
        lut_net->Add(layer_lut2);
        lut_net->Add(layer_lut3);
        lut_net->Add(layer_lut4);
        lut_net->Add(layer_lut5);
        lut_net->Add(bb::BinaryToReal<bb::Bit, float>::Create({1}, mux_size));
        lut_net->SetInputShape(td.x_shape);

        // テーブル化して取り込み(SetInputShape後に取り込みが必要)
        layer_lut0->ImportLayer<float, float>(layer_sl0);
        layer_lut1->ImportLayer<float, float>(layer_sl1);
        layer_lut2->ImportLayer<float, float>(layer_sl2);
        layer_lut3->ImportLayer<float, float>(layer_sl3);
        layer_lut4->ImportLayer<float, float>(layer_sl4);
        layer_lut5->ImportLayer<float, float>(layer_sl5);

        // 評価
        bb::Runner<float>::create_t lut_runner_create;
        lut_runner_create.name           = "DiabetesRegressionBinaryLut";
        lut_runner_create.net            = lut_net;
        lut_runner_create.metricsFunc    = bb::MetricsMeanSquaredError<float>::Create();
        lut_runner_create.print_progress = true;
        auto lut_runner = bb::Runner<float>::Create(lut_runner_create);
        auto lut_accuracy = lut_runner->Evaluation(td, mini_batch_size);
        std::cout << "LUT-Network accuracy : " << lut_accuracy << std::endl;

        {
            // Verilog 出力
            std::string filename = "DiabetesRegressionBinaryLut.v";
            std::ofstream ofs(filename);
            ofs << "`timescale 1ns / 1ps\n\n";
            bb::ExportVerilog_LutModels<>(ofs, "DiabetesRegressionBinaryLut", lut_net);
            std::cout << "export : " << filename << "\n" << std::endl;
        }
    }
}

