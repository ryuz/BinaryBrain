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
#include "bb/ExportVerilog.h"


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
    auto layer_cnv0_mm0 = bb::MicroMlp<>::Create(192);
    auto layer_cnv0_mm1 = bb::MicroMlp<>::Create(32);
    auto layer_cnv1_mm0 = bb::MicroMlp<>::Create(192);
    auto layer_cnv1_mm1 = bb::MicroMlp<>::Create(32);
    auto layer_cnv2_mm0 = bb::MicroMlp<>::Create(192);
    auto layer_cnv2_mm1 = bb::MicroMlp<>::Create(32);
    auto layer_cnv3_mm0 = bb::MicroMlp<>::Create(192);
    auto layer_cnv3_mm1 = bb::MicroMlp<>::Create(32);
    auto layer_mm4 = bb::MicroMlp<>::Create(480);
    auto layer_mm5 = bb::MicroMlp<>::Create(80);

    {
        auto cnv0_sub = bb::Sequential::Create();
        cnv0_sub->Add(layer_cnv0_mm0);
        cnv0_sub->Add(layer_cnv0_mm1);

        auto cnv1_sub = bb::Sequential::Create();
        cnv1_sub->Add(layer_cnv1_mm0);
        cnv1_sub->Add(layer_cnv1_mm1);

        auto cnv2_sub = bb::Sequential::Create();
        cnv2_sub->Add(layer_cnv2_mm0);
        cnv2_sub->Add(layer_cnv2_mm1);

        auto cnv3_sub = bb::Sequential::Create();
        cnv3_sub->Add(layer_cnv3_mm0);
        cnv3_sub->Add(layer_cnv3_mm1);


        auto net = bb::Sequential::Create();
        net->Add(bb::RealToBinary<>::Create(1));
        net->Add(bb::LoweringConvolution<>::Create(cnv0_sub, 3, 3));
        net->Add(bb::LoweringConvolution<>::Create(cnv1_sub, 3, 3));
        net->Add(bb::MaxPooling<>::Create(2, 2));
        net->Add(bb::LoweringConvolution<>::Create(cnv2_sub, 3, 3));
        net->Add(bb::LoweringConvolution<>::Create(cnv3_sub, 3, 3));
        net->Add(bb::MaxPooling<>::Create(2, 2));
        net->Add(layer_mm4);
        net->Add(layer_mm5);
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


    {
        // LUT-network
        auto layer_cnv0_lut0 = bb::BinaryLutN<>::Create(layer_cnv0_mm0->GetOutputShape());
        auto layer_cnv0_lut1 = bb::BinaryLutN<>::Create(layer_cnv0_mm1->GetOutputShape());
        auto layer_cnv1_lut0 = bb::BinaryLutN<>::Create(layer_cnv1_mm0->GetOutputShape());
        auto layer_cnv1_lut1 = bb::BinaryLutN<>::Create(layer_cnv1_mm1->GetOutputShape());
        auto layer_cnv2_lut0 = bb::BinaryLutN<>::Create(layer_cnv2_mm0->GetOutputShape());
        auto layer_cnv2_lut1 = bb::BinaryLutN<>::Create(layer_cnv2_mm1->GetOutputShape());
        auto layer_cnv3_lut0 = bb::BinaryLutN<>::Create(layer_cnv3_mm0->GetOutputShape());
        auto layer_cnv3_lut1 = bb::BinaryLutN<>::Create(layer_cnv3_mm1->GetOutputShape());
        auto layer_lut4      = bb::BinaryLutN<>::Create(layer_mm4->GetOutputShape());
        auto layer_lut5      = bb::BinaryLutN<>::Create(layer_mm5->GetOutputShape());

        // テーブル化して取り込み
        layer_cnv0_lut0->ImportLayer(*layer_cnv0_mm0);
        layer_cnv0_lut1->ImportLayer(*layer_cnv0_mm1);
        layer_cnv1_lut0->ImportLayer(*layer_cnv1_mm0);
        layer_cnv1_lut1->ImportLayer(*layer_cnv1_mm1);
        layer_cnv2_lut0->ImportLayer(*layer_cnv2_mm0);
        layer_cnv2_lut1->ImportLayer(*layer_cnv2_mm1);
        layer_cnv3_lut0->ImportLayer(*layer_cnv3_mm0);
        layer_cnv3_lut1->ImportLayer(*layer_cnv3_mm1);
        layer_lut4     ->ImportLayer(*layer_mm4);
        layer_lut5     ->ImportLayer(*layer_mm5);

        auto cnv0_sub = bb::Sequential::Create();
        cnv0_sub->Add(layer_cnv0_lut0);
        cnv0_sub->Add(layer_cnv0_lut1);

        auto cnv1_sub = bb::Sequential::Create();
        cnv1_sub->Add(layer_cnv1_lut0);
        cnv1_sub->Add(layer_cnv1_lut1);

        auto cnv2_sub = bb::Sequential::Create();
        cnv2_sub->Add(layer_cnv2_lut0);
        cnv2_sub->Add(layer_cnv2_lut1);

        auto cnv3_sub = bb::Sequential::Create();
        cnv3_sub->Add(layer_cnv3_lut0);
        cnv3_sub->Add(layer_cnv3_lut1);

        auto cnv0 = bb::LoweringConvolution<bb::Bit>::Create(cnv0_sub, 3, 3);
        auto cnv1 = bb::LoweringConvolution<bb::Bit>::Create(cnv1_sub, 3, 3);
        auto cnv2 = bb::LoweringConvolution<bb::Bit>::Create(cnv2_sub, 3, 3);
        auto cnv3 = bb::LoweringConvolution<bb::Bit>::Create(cnv3_sub, 3, 3);

        auto lut_net = bb::Sequential::Create();
        lut_net->Add(bb::RealToBinary<float, bb::Bit>::Create(1));
        lut_net->Add(cnv0);
        lut_net->Add(cnv1);
        lut_net->Add(bb::MaxPooling<>::Create(2, 2));
        lut_net->Add(cnv2);
        lut_net->Add(cnv3);
        lut_net->Add(bb::MaxPooling<>::Create(2, 2));
        lut_net->Add(layer_lut4);
        lut_net->Add(layer_lut5);
        lut_net->Add(bb::BinaryToReal<bb::Bit, float>::Create({ 10 }, 1));
        lut_net->SetInputShape({28, 28, 1});
        
        // 評価
        bb::Runner<float>::create_t lut_runner_create;
        lut_runner_create.name      = "Lut_MnistSimpleLutCnn";
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
            std::vector< std::shared_ptr< bb::LoweringConvolution<bb::Bit> > >  vec_cnv0;
            std::vector< std::shared_ptr< bb::LoweringConvolution<bb::Bit> > >  vec_cnv1;
            std::vector< std::shared_ptr< bb::LutLayer<> > >                    vec_mlp;

            vec_cnv0.push_back(cnv0);
            vec_cnv0.push_back(cnv1);
            vec_cnv1.push_back(cnv2);
            vec_cnv1.push_back(cnv3);
            vec_mlp.push_back(layer_lut4);
            vec_mlp.push_back(layer_lut5);

            std::string filename = "verilog/MnistSimpleLutCnn.v";
            std::ofstream ofs(filename);
            bb::ExportVerilog_LutCnnLayersAxi4s(ofs, "MnistSimpleLutCnnCnv0", vec_cnv0);
            bb::ExportVerilog_LutCnnLayersAxi4s(ofs, "MnistSimpleLutCnnCnv1", vec_cnv1);
            bb::ExportVerilog_LutLayers(ofs, "MnistSimpleLutCnnCnv1", vec_mlp);
            std::cout << "export : " << filename << "\n" << std::endl;
        }
    }
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
