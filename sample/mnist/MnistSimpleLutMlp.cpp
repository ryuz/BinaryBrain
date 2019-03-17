// --------------------------------------------------------------------------
//  BinaryBrain  -- binary network evaluation platform
//   MNIST sample
//
//                                Copyright (C) 2018-2019 by Ryuji Fuchikami
// --------------------------------------------------------------------------


#include <iostream>
#include <fstream>
#include <numeric>
#include <random>
#include <chrono>

#include "bb/RealToBinary.h"
#include "bb/BinaryToReal.h"
#include "bb/MicroMlp.h"
#include "bb/BinaryLutN.h"
#include "bb/BatchNormalization.h"
#include "bb/ReLU.h"
#include "bb/LossSoftmaxCrossEntropy.h"
#include "bb/AccuracyCategoricalClassification.h"
#include "bb/OptimizerAdam.h"
#include "bb/OptimizerSgd.h"
#include "bb/LoadMnist.h"
#include "bb/ShuffleSet.h"
#include "bb/Utility.h"
#include "bb/Sequential.h"
#include "bb/Runner.h"
#include "bb/ExportVerilog.h"


// MNIST CNN with LUT networks
void MnistSimpleLutMlp(int epoch_size, size_t mini_batch_size, bool binary_mode)
{
  // load MNIST data
#ifdef _DEBUG
	auto td = bb::LoadMnist<>::Load(10, 64, 32);
#else
    auto td = bb::LoadMnist<>::Load(10);
#endif

#ifdef _DEBUG
    auto layer_mm0 = bb::MicroMlp<>::Create({16});
    auto layer_mm1 = bb::MicroMlp<>::Create({16});
    auto layer_mm2 = bb::MicroMlp<>::Create({30});
#else
    auto layer_mm0 = bb::MicroMlp<>::Create({1024});
    auto layer_mm1 = bb::MicroMlp<>::Create({360});
    auto layer_mm2 = bb::MicroMlp<>::Create({60});
#endif

    {
        auto net = bb::Sequential::Create();
        net->Add(bb::RealToBinary<>::Create(7));
        net->Add(layer_mm0);
        net->Add(layer_mm1);
        net->Add(layer_mm2);
        net->Add(bb::BinaryToReal<float, float>::Create({10}, 7));
        net->SetInputShape(td.x_shape);

        if ( binary_mode ) {
            net->SendCommand("binary true");
            std::cout << "binary mode" << std::endl;
        }

    //  net->SendCommand("host_only true", "BatchNormalization");

        net->PrintInfo();

        // fitting
        bb::Runner<float>::create_t runner_create;
        runner_create.name      = "MnistSimpleLutMlp";
        runner_create.net       = net;
        runner_create.lossFunc  = bb::LossSoftmaxCrossEntropy<float>::Create();
        runner_create.accFunc   = bb::AccuracyCategoricalClassification<float>::Create(10);
        runner_create.optimizer = bb::OptimizerAdam<float>::Create();
        runner_create.initial_evaluation = false;
        runner_create.print_progress = true;
        auto runner = bb::Runner<float>::Create(runner_create);
        runner->Fitting(td, epoch_size, mini_batch_size);
    }

    {
        // LUT-network
        auto layer_lut0 = bb::BinaryLutN<>::Create(layer_mm0->GetOutputShape());
        auto layer_lut1 = bb::BinaryLutN<>::Create(layer_mm1->GetOutputShape());
        auto layer_lut2 = bb::BinaryLutN<>::Create(layer_mm2->GetOutputShape());
        auto lut_net = bb::Sequential::Create();
        lut_net->Add(bb::RealToBinary<float, bb::Bit>::Create(7));
        lut_net->Add(layer_lut0);
        lut_net->Add(layer_lut1);
        lut_net->Add(layer_lut2);
        lut_net->Add(bb::BinaryToReal<bb::Bit, float>::Create({10}, 7));
        lut_net->SetInputShape(td.x_shape);

        layer_lut0->ImportLayer(*layer_mm0);
        layer_lut1->ImportLayer(*layer_mm1);
        layer_lut2->ImportLayer(*layer_mm2);

        bb::Runner<float>::create_t lut_runner_create;
        lut_runner_create.name      = "Lut_MnistSimpleLutMlp";
        lut_runner_create.net       = lut_net;
        lut_runner_create.lossFunc  = bb::LossSoftmaxCrossEntropy<float>::Create();
        lut_runner_create.accFunc   = bb::AccuracyCategoricalClassification<float>::Create(10);
        lut_runner_create.optimizer = bb::OptimizerAdam<float>::Create();
        lut_runner_create.initial_evaluation = false;
        lut_runner_create.print_progress = true;
        auto lut_runner = bb::Runner<float>::Create(lut_runner_create);
        auto lut_accuracy = lut_runner->Evaluation(td, mini_batch_size);
        std::cout << "lut_accuracy : " << lut_accuracy << std::endl;

        {
            // Verilog 出力
            std::string filename = "MnistSimpleLutMlp.v";
            std::ofstream ofs(filename);
            bb::ExportVerilog_LutLayer(ofs, "layer0", *layer_lut0);
            bb::ExportVerilog_LutLayer(ofs, "layer1", *layer_lut1);
            bb::ExportVerilog_LutLayer(ofs, "layer2", *layer_lut2);
            std::cout << "export : " << filename << "\n" << std::endl;
        }
    }
}

