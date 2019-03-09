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

#include "bb/DenseAffine.h"
#include "bb/MicroMlp.h"
#include "bb/BatchNormalization.h"
#include "bb/ReLU.h"
#include "bb/AccuracyMeanSquaredError.h"
#include "bb/LossMeanSquaredError.h"
#include "bb/OptimizerAdam.h"
#include "bb/OptimizerSgd.h"
#include "bb/LoadMnist.h"
#include "bb/ShuffleSet.h"
#include "bb/Utility.h"
#include "bb/Sequential.h"
#include "bb/Runner.h"
#include "bb/RealToBinary.h"
#include "bb/BinaryToReal.h"


template<typename T=float>
bb::TrainData<T> LoadDiabetes(int num_train=400)
{
	const int n = 442;

	std::ifstream ifs_x("diabetes_data.txt");
	std::ifstream ifs_t("diabetes_target.txt");

	bb::TrainData<T> td;
	td.x_shape = bb::indices_t({ 10 });
	td.t_shape = bb::indices_t({ 1 });

	for (int i = 0; i < num_train; ++i) {
		std::vector<T> train(10);
		std::vector<T> target(1);
		for (int j = 0; j < 10; ++j) {
			ifs_x >> train[j];
		}
		ifs_t >> target[0];

		td.x_train.push_back(train);
		td.t_train.push_back(target);
	}

	for (int i = 0; i < n - num_train; ++i) {
		std::vector<T> train(10);
		std::vector<T> target(1);
		for (int j = 0; j < 10; ++j) {
			ifs_x >> train[j];
		}
		ifs_t >> target[0];

		td.x_test.push_back(train);
		td.t_test.push_back(target);
	}

	return td;
}



void DiabetesRegressionBinaryLut(int epoch_size, size_t mini_batch_size, size_t mux_size)
{
	// load diabetes data
	auto td = LoadDiabetes<>();

	bb::TrainDataNormalize(td);

    auto net = bb::Sequential::Create();
	net->Add(bb::RealToBinary<>::Create(mux_size));
	net->Add(bb::MicroMlp<6, 16>::Create({ 512 }));
	net->Add(bb::MicroMlp<6, 16>::Create({ 216 }));
	net->Add(bb::MicroMlp<6, 16>::Create({ 36 }));
    net->Add(bb::MicroMlp<6, 16>::Create({ 6 }));
    net->Add(bb::MicroMlp<6, 16>::Create({ 1 }));
	net->Add(bb::BinaryToReal<>::Create({ 1 }, mux_size));

	net->SetInputShape({ 10 });

    bb::FrameBuffer x(BB_TYPE_FP32, mini_batch_size, { 10 });
	bb::FrameBuffer t(BB_TYPE_FP32, mini_batch_size, { 1 });

    bb::Runner<float>::create_t runner_create;
    runner_create.name      = "DiabetesRegressionBinaryLut";
    runner_create.net       = net;
    runner_create.lossFunc  = bb::LossMeanSquaredError<float>::Create();
    runner_create.accFunc   = bb::AccuracyMeanSquaredError<float>::Create();
	runner_create.optimizer = bb::OptimizerSgd<float>::Create(0.00001f);
//	runner_create.optimizer = bb::OptimizerAdam<float>::Create();
    runner_create.serial_write = false;
	runner_create.over_write = true;
	runner_create.print_progress = false;
    runner_create.initial_evaluation = true;
    auto runner = bb::Runner<float>::Create(runner_create);

    runner->Fitting(td, epoch_size, mini_batch_size);
}

