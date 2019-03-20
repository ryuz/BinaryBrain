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
#include "bb/LoweringConvolution.h"
#include "bb/BatchNormalization.h"
#include "bb/ReLU.h"
#include "bb/MaxPooling.h"
#include "bb/LossSoftmaxCrossEntropy.h"
#include "bb/AccuracyCategoricalClassification.h"
#include "bb/OptimizerAdam.h"
#include "bb/OptimizerSgd.h"
#include "bb/LoadMnist.h"
#include "bb/ShuffleSet.h"
#include "bb/Utility.h"
#include "bb/Sequential.h"
#include "bb/Runner.h"


// MNIST CNN with LUT networks
void MnistSimpleLutCnn(int epoch_size, size_t mini_batch_size, bool binary_mode)
{
  // load MNIST data
#ifdef _DEBUG
	auto td = bb::LoadMnist<>::Load(10, 512, 128);
    std::cout << "!!! debug mode !!!" << std::endl;
#else
    auto td = bb::LoadMnist<>::Load(10);
#endif

    // create network
    auto cnv0_sub = bb::Sequential::Create();
    cnv0_sub->Add(bb::MicroMlp<>::Create(192));
    cnv0_sub->Add(bb::MicroMlp<>::Create(32));

    auto cnv1_sub = bb::Sequential::Create();
    cnv1_sub->Add(bb::MicroMlp<>::Create(192));
    cnv1_sub->Add(bb::MicroMlp<>::Create(32));

    auto cnv2_sub = bb::Sequential::Create();
    cnv2_sub->Add(bb::MicroMlp<>::Create(192));
    cnv2_sub->Add(bb::MicroMlp<>::Create(32));

    auto cnv3_sub = bb::Sequential::Create();
    cnv3_sub->Add(bb::MicroMlp<>::Create(192));
    cnv3_sub->Add(bb::MicroMlp<>::Create(32));

    auto net = bb::Sequential::Create();
    net->Add(bb::RealToBinary<>::Create(1));
    net->Add(bb::LoweringConvolution<>::Create(cnv0_sub, 3, 3));
    net->Add(bb::LoweringConvolution<>::Create(cnv1_sub, 3, 3));
    net->Add(bb::MaxPooling<>::Create(2, 2));
    net->Add(bb::LoweringConvolution<>::Create(cnv2_sub, 3, 3));
    net->Add(bb::LoweringConvolution<>::Create(cnv3_sub, 3, 3));
    net->Add(bb::MaxPooling<>::Create(2, 2));
    net->Add(bb::MicroMlp<>::Create({480}));
    net->Add(bb::MicroMlp<>::Create({80}));
    net->Add(bb::BinaryToReal<>::Create({ 10 }, 1));
    net->SetInputShape({28, 28, 1});

    if ( binary_mode ) {
        std::cout << "binary mode" << std::endl;
        net->SendCommand("binary true");
    }

    // print model information
    net->PrintInfo();

    // run fitting
    bb::Runner<float>::create_t runner_create;
    runner_create.name      = "MnistSimpleLutCnn";
    runner_create.net       = net;
    runner_create.lossFunc  = bb::LossSoftmaxCrossEntropy<float>::Create();
    runner_create.accFunc   = bb::AccuracyCategoricalClassification<float>::Create(10);
    runner_create.optimizer = bb::OptimizerAdam<float>::Create();
    runner_create.print_progress = true;
    auto runner = bb::Runner<float>::Create(runner_create);
    runner->Fitting(td, epoch_size, mini_batch_size);
}


// RTL simulation 用データの出力
static void WriteTestImage(void)
{
	// load MNIST data
	auto td = bb::LoadMnist<>::Load();

	const int w = 640 / 4;
	const int h = 480 / 4;

	unsigned char img[h][w];
	for (int y = 0; y < h; ++y) {
		for (int x = 0; x < w; ++x) {
			int idx = (y / 28) * (w / 28) + (x / 28);
			int xx = x % 28;
			int yy = y % 28;
			img[y][x] = (unsigned char)(td.x_test[idx][yy * 28 + xx] * 255.0f);
		}
	}

	{
		std::ofstream ofs("mnist_test.pgm");
		ofs << "P2" << std::endl;
		ofs << w << " " << h << std::endl;
		ofs << "255" << std::endl;
		for (int y = 0; y < h; ++y) {
			for (int x = 0; x < w; ++x) {
				ofs << (int)img[y][x] << std::endl;
			}
		}
	}

	{
		std::ofstream ofs("mnist_test.ppm");
		ofs << "P3" << std::endl;
		ofs << w << " " << h << std::endl;
		ofs << "255" << std::endl;
		for (int y = 0; y < h; ++y) {
			for (int x = 0; x < w; ++x) {
				ofs << (int)img[y][x] << " " << (int)img[y][x] << " " << (int)img[y][x] << std::endl;
			}
		}
	}
}



// end of file
