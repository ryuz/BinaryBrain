// --------------------------------------------------------------------------
//  BinaryBrain  -- binary network evaluation platform
//   diabetes regression sample
//
//                                Copyright (C) 2018-2019 by Ryuji Fuchikami
// --------------------------------------------------------------------------


#include <iostream>

#include "bb/Sequential.h"
#include "bb/DenseAffine.h"
#include "bb/MicroMlp.h"
#include "bb/BinaryLutN.h"
#include "bb/BatchNormalization.h"
#include "bb/ReLU.h"
#include "bb/MetricsMeanSquaredError.h"
#include "bb/LossMeanSquaredError.h"
#include "bb/OptimizerAdam.h"
#include "bb/OptimizerSgd.h"
#include "bb/RealToBinary.h"
#include "bb/BinaryToReal.h"
#include "bb/UniformDistributionGenerator.h"
#include "bb/Runner.h"
#include "bb/ExportVerilog.h"

#include "LoadDiabetes.h"


void DiabetesRegressionMicroMlpLut(int epoch_size, size_t mini_batch_size, size_t mux_size)
{
    // load diabetes data
    auto td = LoadDiabetes<>();
    bb::TrainDataNormalize(td);

    auto layer_mm0 = bb::MicroMlp<6, 16>::Create(1024);
    auto layer_mm1 = bb::MicroMlp<6, 16>::Create(512);
    auto layer_mm2 = bb::MicroMlp<6, 16>::Create(216);
    auto layer_mm3 = bb::MicroMlp<6, 16>::Create(36);
    auto layer_mm4 = bb::MicroMlp<6, 16>::Create(6);
    auto layer_mm5 = bb::MicroMlp<6, 16>::Create(1);

    {
        // uMLPで学習
        auto net = bb::Sequential::Create();
        net->Add(bb::RealToBinary<>::Create(mux_size, bb::UniformDistributionGenerator<float>::Create(0.0f, 1.0f, 1)));
        net->Add(layer_mm0);
        net->Add(layer_mm1);
        net->Add(layer_mm2);
        net->Add(layer_mm3);
        net->Add(layer_mm4);
        net->Add(layer_mm5);
        net->Add(bb::BinaryToReal<>::Create(mux_size, td.t_shape));
        net->SetInputShape(td.x_shape);

        net->SendCommand("binary true");

        bb::Runner<float>::create_t runner_create;
        runner_create.name               = "DiabetesRegressionMicroMlpLut";
        runner_create.net                = net;
        runner_create.lossFunc           = bb::LossMeanSquaredError<float>::Create();
        runner_create.metricsFunc        = bb::MetricsMeanSquaredError<float>::Create();
        runner_create.optimizer          = bb::OptimizerAdam<float>::Create();
        runner_create.file_read          = false;
        runner_create.file_write         = true;
        runner_create.print_progress     = false;
        runner_create.initial_evaluation = true;
        auto runner = bb::Runner<float>::Create(runner_create);

        runner->Fitting(td, epoch_size, mini_batch_size);
    }
    
    {
        // LUT-network
        auto layer_lut0 = bb::BinaryLutN<>::Create(layer_mm0->GetOutputShape());
        auto layer_lut1 = bb::BinaryLutN<>::Create(layer_mm1->GetOutputShape());
        auto layer_lut2 = bb::BinaryLutN<>::Create(layer_mm2->GetOutputShape());
        auto layer_lut3 = bb::BinaryLutN<>::Create(layer_mm3->GetOutputShape());
        auto layer_lut4 = bb::BinaryLutN<>::Create(layer_mm4->GetOutputShape());
        auto layer_lut5 = bb::BinaryLutN<>::Create(layer_mm5->GetOutputShape());

        auto lut_net = bb::Sequential::Create();
        lut_net->Add(bb::RealToBinary<bb::Bit>::Create(mux_size, bb::UniformDistributionGenerator<float>::Create(0.0f, 1.0f, 1)));
        lut_net->Add(layer_lut0);
        lut_net->Add(layer_lut1);
        lut_net->Add(layer_lut2);
        lut_net->Add(layer_lut3);
        lut_net->Add(layer_lut4);
        lut_net->Add(layer_lut5);
        lut_net->Add(bb::BinaryToReal<bb::Bit>::Create(mux_size, td.t_shape));
        lut_net->SetInputShape(td.x_shape);

        // テーブル化して取り込み(SetInputShape後に取り込みが必要)
        std::cout << "parameter copy to LUT-Network" << std::endl;
        layer_lut0->ImportLayer(layer_mm0);
        layer_lut1->ImportLayer(layer_mm1);
        layer_lut2->ImportLayer(layer_mm2);
        layer_lut3->ImportLayer(layer_mm3);
        layer_lut4->ImportLayer(layer_mm4);
        layer_lut5->ImportLayer(layer_mm5);

        // 評価
        bb::Runner<float>::create_t lut_runner_create;
        lut_runner_create.name           = "DiabetesRegressionBinaryLut";
        lut_runner_create.net            = lut_net;
        lut_runner_create.lossFunc       = bb::LossMeanSquaredError<float>::Create();
        lut_runner_create.metricsFunc    = bb::MetricsMeanSquaredError<float>::Create();
        lut_runner_create.optimizer      = bb::OptimizerAdam<float>::Create();
        lut_runner_create.print_progress = true;
        auto lut_runner = bb::Runner<float>::Create(lut_runner_create);
        auto lut_accuracy = lut_runner->Evaluation(td, mini_batch_size);
        std::cout << "LUT-Network accuracy : " << lut_accuracy << std::endl;

        {
            // Verilog 出力
            std::string filename = "DiabetesRegressionBinaryLut.v";
            std::ofstream ofs(filename);
            ofs << "`timescale 1ns / 1ps\n\n";
            bb::ExportVerilog_LutModels(ofs, "DiabetesRegressionBinaryLut", lut_net);
            std::cout << "export : " << filename << "\n" << std::endl;
        }
    }
}

