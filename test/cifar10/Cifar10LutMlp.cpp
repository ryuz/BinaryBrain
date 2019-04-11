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
#include "bb/MetricsCategoricalAccuracy.h"
#include "bb/OptimizerAdam.h"
#include "bb/OptimizerSgd.h"
#include "bb/LoadCifar10.h"
#include "bb/ShuffleSet.h"
#include "bb/Utility.h"
#include "bb/Sequential.h"
#include "bb/Runner.h"
#include "bb/ExportVerilog.h"


// static void WriteMnistDataFile(std::string train_file, std::string test_file, int train_size, int test_size);


// MNIST CNN with LUT networks
void Cifar10LutMlp(int epoch_size, size_t mini_batch_size, bool binary_mode)
{
    std::string net_name = "Cifar10LutMlp";
    int const mux_size = 7;

  // load cifar-10 data
#ifdef _DEBUG
	auto td = bb::LoadCifar10<>::Load(1);
    std::cout << "!!! debug mode !!!" << std::endl;
#else
    auto td = bb::LoadCifar10<>::Load();
#endif

#ifdef _DEBUG
    auto layer_mm0 = bb::MicroMlp<>::Create({16});
    auto layer_mm1 = bb::MicroMlp<>::Create({16});
    auto layer_mm2 = bb::MicroMlp<>::Create({10});
#else
    auto layer_mm0 = bb::MicroMlp<>::Create({1024});
    auto layer_mm1 = bb::MicroMlp<>::Create({480});
    auto layer_mm2 = bb::MicroMlp<>::Create({80});
#endif

    {
        auto net = bb::Sequential::Create();
        net->Add(bb::RealToBinary<>::Create(mux_size));
        net->Add(layer_mm0);
        net->Add(layer_mm1);
        net->Add(layer_mm2);
        net->Add(bb::BinaryToReal<float, float>::Create({10}, mux_size));
        net->SetInputShape(td.x_shape);

        if ( binary_mode ) {
            net->SendCommand("binary true");
            std::cout << "binary mode" << std::endl;
        }

    //  net->SendCommand("host_only true", "BatchNormalization");

        net->PrintInfo();

        // fitting
        bb::Runner<float>::create_t runner_create;
        runner_create.name           = net_name;
        runner_create.net            = net;
        runner_create.lossFunc       = bb::LossSoftmaxCrossEntropy<float>::Create();
        runner_create.metricsFunc    = bb::MetricsCategoricalAccuracy<float>::Create();
        runner_create.optimizer      = bb::OptimizerAdam<float>::Create();
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
        lut_net->Add(bb::RealToBinary<float, bb::Bit>::Create());
        lut_net->Add(layer_lut0);
        lut_net->Add(layer_lut1);
        lut_net->Add(layer_lut2);
        lut_net->Add(bb::BinaryToReal<bb::Bit, float>::Create({10}));
        lut_net->SetInputShape(td.x_shape);

        // テーブル化して取り込み(SetInputShape後に取り込みが必要)
        layer_lut0->ImportLayer<float, float>(layer_mm0);
        layer_lut1->ImportLayer<float, float>(layer_mm1);
        layer_lut2->ImportLayer<float, float>(layer_mm2);

        // 評価
        bb::Runner<float>::create_t lut_runner_create;
        lut_runner_create.name           = "Lut_" + net_name;
        lut_runner_create.net            = lut_net;
        lut_runner_create.lossFunc       = bb::LossSoftmaxCrossEntropy<float>::Create();
        lut_runner_create.metricsFunc    = bb::MetricsCategoricalAccuracy<float>::Create();
        lut_runner_create.optimizer      = bb::OptimizerAdam<float>::Create();
        lut_runner_create.print_progress = true;
        auto lut_runner = bb::Runner<float>::Create(lut_runner_create);
        auto lut_accuracy = lut_runner->Evaluation(td, mini_batch_size);
        std::cout << "lut_accuracy : " << lut_accuracy << std::endl;

        {
            // Verilog 出力
            std::string filename = "verilog/" + net_name + ".v";
            std::ofstream ofs(filename);
            ofs << "`timescale 1ns / 1ps\n\n";
            bb::ExportVerilog_LutLayers<>(ofs, net_name, lut_net);
            std::cout << "export : " << filename << "\n" << std::endl;

            // RTL simulation 用データの出力
//          WriteMnistDataFile("verilog/mnist_train.txt", "verilog/mnist_test.txt", 60000, 10000);
        }
    }
}


/*
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


*/
