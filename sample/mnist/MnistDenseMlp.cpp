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

#include "bb/DenseAffine.h"
#include "bb/BatchNormalization.h"
#include "bb/ReLU.h"
#include "bb/LossSoftmaxCrossEntropy.h"
#include "bb/MetricsCategoricalAccuracy.h"
#include "bb/OptimizerAdam.h"
#include "bb/OptimizerSgd.h"
#include "bb/LoadMnist.h"
#include "bb/ShuffleSet.h"
#include "bb/Utility.h"
#include "bb/Sequential.h"
#include "bb/Runner.h"



void MnistDenseMlp(int epoch_size, int mini_batch_size, int max_run_size, bool binary_mode, bool file_read)
{
    // load MNIST data
#ifdef _DEBUG
    auto td = bb::LoadMnist<>::Load(10, 512, 128);
    std::cout << "!!! debug mode !!!" << std::endl;
#else
    auto td = bb::LoadMnist<>::Load(10);
#endif

    // create network
    auto net = bb::Sequential::Create();
    net->Add(bb::DenseAffine<float>::Create(256));
    net->Add(bb::ReLU<float>::Create());
    net->Add(bb::DenseAffine<float>::Create(td.t_shape));
    net->SetInputShape(td.x_shape);

    std::cout << "binary_mode : " << binary_mode << std::endl;
    if ( binary_mode ) {
        net->SendCommand("binary true");
    }
    
    // run fitting
    bb::Runner<float>::create_t runner_create;
    runner_create.name               = "MnistDenseMlp";
    runner_create.net                = net;
    runner_create.lossFunc           = bb::LossSoftmaxCrossEntropy<float>::Create();
    runner_create.metricsFunc        = bb::MetricsCategoricalAccuracy<float>::Create();
    runner_create.optimizer          = bb::OptimizerAdam<float>::Create();
    runner_create.max_run_size       = max_run_size;    // ���ۂ�1��̎��s�T�C�Y
    runner_create.file_read          = file_read;       // �O�̌v�Z���ʂ�����Γǂݍ���ōĊJ���邩
    runner_create.file_write         = true;            // �v�Z���ʂ��t�@�C���ɕۑ����邩
    runner_create.print_progress     = true;            // �r�����ʂ�\��
    runner_create.initial_evaluation = file_read;       // �t�@�C����ǂ񂾏ꍇ�͍ŏ��ɕ]�����Ă���
    auto runner = bb::Runner<float>::Create(runner_create);
    runner->Fitting(td, epoch_size, mini_batch_size);
}

