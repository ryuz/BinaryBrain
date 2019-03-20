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


static void WriteMnistDataFile(std::string train_file, std::string test_file, int train_size, int test_size);

// MNIST CNN with LUT networks
void MnistSimpleLutMlp(int epoch_size, size_t mini_batch_size, bool binary_mode)
{
    // RTL simulation 用データの出力
    WriteMnistDataFile("verilog/_train.txt", "verilog/mnist_test.txt", 60000, 10000);


  // load MNIST data
#ifdef _DEBUG
	auto td = bb::LoadMnist<>::Load(10, 64, 32);
    std::cout << "!!! debug mode !!!" << std::endl;
#else
    auto td = bb::LoadMnist<>::Load(10);
#endif

#ifdef _DEBUG
    auto layer_mm0 = bb::MicroMlp<>::Create({16});
    auto layer_mm1 = bb::MicroMlp<>::Create({16});
    auto layer_mm2 = bb::MicroMlp<>::Create({16});
    auto layer_mm3 = bb::MicroMlp<>::Create({30});
#else
    auto layer_mm0 = bb::MicroMlp<>::Create({1024});
    auto layer_mm1 = bb::MicroMlp<>::Create({360});
    auto layer_mm2 = bb::MicroMlp<>::Create({60});
    auto layer_mm3 = bb::MicroMlp<>::Create({10});
#endif

    {
        auto net = bb::Sequential::Create();
        net->Add(bb::RealToBinary<>::Create(7));
        net->Add(layer_mm0);
        net->Add(layer_mm1);
        net->Add(layer_mm2);
        net->Add(layer_mm3);
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
        auto layer_lut3 = bb::BinaryLutN<>::Create(layer_mm3->GetOutputShape());
        auto lut_net = bb::Sequential::Create();
        lut_net->Add(bb::RealToBinary<float, bb::Bit>::Create(7));
        lut_net->Add(layer_lut0);
        lut_net->Add(layer_lut1);
        lut_net->Add(layer_lut2);
        lut_net->Add(layer_lut3);
        lut_net->Add(bb::BinaryToReal<bb::Bit, float>::Create({10}, 7));
        lut_net->SetInputShape(td.x_shape);

        // テーブル化して取り込み
        layer_lut0->ImportLayer(*layer_mm0);
        layer_lut1->ImportLayer(*layer_mm1);
        layer_lut2->ImportLayer(*layer_mm2);
        layer_lut3->ImportLayer(*layer_mm3);

        // 評価
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
            std::string filename = "verilog/MnistSimpleLutMlp.v";
            std::ofstream ofs(filename);
            bb::ExportVerilog_LutLayers<>(ofs, "MnistSimpleLutMlp", lut_net);
            std::cout << "export : " << filename << "\n" << std::endl;
        }
    }
}





static void WriteMnistDataFile(std::ostream& ofs, std::vector< std::vector<float> > x, std::vector< std::vector<float> > y)
{
	for (size_t i = 0; i < x.size(); ++i) {
		auto yi = bb::argmax<>(y[i]);

		for (int j = 7; j >= 0; --j) {
			ofs << ((yi >> j) & 1);
		}
		ofs << "_";

		for (int j = 28*28-1; j >= 0; --j) {
			if (x[i][j] > 0.5f) {
				ofs << "1";
			}
			else {
				ofs << "0";
			}
		}
		ofs << std::endl;
	}
}


// write data file for verilog testbench
static void WriteMnistDataFile(std::string train_file, std::string test_file, int train_size, int test_size)
{
	// load MNIST data
	auto td = bb::LoadMnist<>::Load(10, train_size, test_size);

	// write train data
	{
		std::ofstream ofs_train(train_file);
		WriteMnistDataFile(ofs_train, td.x_train, td.t_train);
	}

	// write test data
	{
		std::ofstream ofs_test(test_file);
		WriteMnistDataFile(ofs_test, td.x_test, td.t_test);
	}
}


