// --------------------------------------------------------------------------
//  BinaryBrain  -- binary network evaluation platform
//   MNIST sample
//
//                                Copyright (C) 2018-2019 by Ryuji Fuchikami
// --------------------------------------------------------------------------


#include <iostream>

#include "bb/Sequential.h"
#include "bb/DenseAffine.h"
#include "bb/DepthwiseDenseAffine.h"
#include "bb/SparseLutN.h"
#include "bb/BatchNormalization.h"
#include "bb/ReLU.h"
#include "bb/Shuffle.h"
#include "bb/LoweringConvolution.h"
#include "bb/MaxPooling.h"
#include "bb/BinaryModulation.h"
#include "bb/OptimizerAdam.h"
#include "bb/LossSoftmaxCrossEntropy.h"
#include "bb/MetricsCategoricalAccuracy.h"
#include "bb/Runner.h"
#include "bb/LoadCifar10.h"


template<typename T>
std::shared_ptr<bb::Model> MakeConvLayer(bb::indices_t shape)
{
    auto net = bb::Sequential::Create();
    net->Add(bb::DenseAffine<float>::Create(shape));
    net->Add(bb::BatchNormalization<float>::Create());
    net->Add(bb::ReLU<T>::Create());
    return net;
}

template<typename T>
std::shared_ptr<bb::Model> MakePointwiseLayer(bb::indices_t shape, bb::index_t unit)
{
    auto net = bb::Sequential::Create();
    net->Add(bb::DepthwiseDenseAffine<float>::Create(shape, unit));
    net->Add(bb::BatchNormalization<float>::Create());
    net->Add(bb::ReLU<T>::Create());
    return net;
}

template<typename T>
std::shared_ptr<bb::Model> MakeDepthwiseLayer(bb::indices_t shape)
{
#if 0
    auto net = bb::Sequential::Create();
    net->Add(bb::DepthwiseDenseAffine<float>::Create(shape));
    net->Add(bb::BatchNormalization<float>::Create());
    net->Add(bb::ReLU<T>::Create());
    return net;
#else
    return bb::SparseLutN<6, T>::Create(shape, true, "depthwise");
#endif
}


template<typename T>
void Cifar10ShuffleNet_(int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read)
{
    std::string net_name = "Cifar10ShuffleNet";


  // load cifar-10 data
#ifdef _DEBUG
    auto td = bb::LoadCifar10<>::Load(1);
    std::cout << "!!! debug mode !!!" << std::endl;
#else
    auto td = bb::LoadCifar10<>::Load();
#endif

    int w            = 4;
    int shuffle_unit = 32;

    {
        std::cout << "\n<Training>" << std::endl;

        // create network
        auto main_net = bb::Sequential::Create();

        main_net->Add(bb::RealToBinary<T>::Create(train_modulation_size, 16));

        main_net->Add(bb::LoweringConvolution<T>::Create(MakePointwiseLayer<T>({1, 1,  32*w}, shuffle_unit), 1, 1));
//      main_net->Add(bb::Shuffle::Create                                     ({1, 1,  32*w}, shuffle_unit));
        main_net->Add(bb::LoweringConvolution<T>::Create(MakeDepthwiseLayer<T>({1, 1,  32*w}              ), 3, 3));  // 30x30
        main_net->Add(bb::LoweringConvolution<T>::Create(MakePointwiseLayer<T>({1, 1,  32  }, shuffle_unit), 1, 1));

        main_net->Add(bb::LoweringConvolution<T>::Create(MakePointwiseLayer<T>({1,   1, 64*w}, shuffle_unit), 1, 1));
        main_net->Add(bb::Shuffle::Create                                     ({30, 30, 64*w}, shuffle_unit));
        main_net->Add(bb::LoweringConvolution<T>::Create(MakeDepthwiseLayer<T>({1,   1, 64*w}              ), 3, 3));  // 28x28
        main_net->Add(bb::LoweringConvolution<T>::Create(MakePointwiseLayer<T>({1,   1, 64  }, shuffle_unit), 1, 1));

        main_net->Add(bb::MaxPooling<T>::Create(2, 2));                                                               // 14x14

        main_net->Add(bb::LoweringConvolution<T>::Create(MakePointwiseLayer<T>({1, 1,  64*w}, shuffle_unit), 1, 1));
        main_net->Add(bb::Shuffle::Create                                     ({14, 14,  64*w}, shuffle_unit));
        main_net->Add(bb::LoweringConvolution<T>::Create(MakeDepthwiseLayer<T>({1, 1,  64*w}              ), 3, 3));  // 12x12
        main_net->Add(bb::LoweringConvolution<T>::Create(MakePointwiseLayer<T>({1, 1,  64  }, shuffle_unit), 1, 1));

        main_net->Add(bb::LoweringConvolution<T>::Create(MakePointwiseLayer<T>({1, 1, 128*w}, shuffle_unit), 1, 1));
        main_net->Add(bb::Shuffle::Create                                     ({12, 12, 128*w}, shuffle_unit));
        main_net->Add(bb::LoweringConvolution<T>::Create(MakeDepthwiseLayer<T>({1, 1, 128*w}              ), 3, 3));  // 10x10
        main_net->Add(bb::LoweringConvolution<T>::Create(MakePointwiseLayer<T>({1, 1, 128  }, shuffle_unit), 1, 1));

        main_net->Add(bb::MaxPooling<T>::Create(2, 2));                                                               // 5x5 

        main_net->Add(bb::LoweringConvolution<T>::Create(MakePointwiseLayer<T>({1, 1, 128*w}, shuffle_unit), 1, 1));
        main_net->Add(bb::Shuffle::Create                                     ({5, 5, 128*w}, shuffle_unit));
        main_net->Add(bb::LoweringConvolution<T>::Create(MakeDepthwiseLayer<T>({1, 1, 128*w}              ), 3, 3));  // 3x3
        main_net->Add(bb::LoweringConvolution<T>::Create(MakePointwiseLayer<T>({1, 1, 128  }, shuffle_unit), 1, 1));

        main_net->Add(bb::LoweringConvolution<T>::Create(MakePointwiseLayer<T>({1, 1, 256*w}, shuffle_unit), 1, 1));
        main_net->Add(bb::Shuffle::Create                                     ({3, 3, 256*w}, shuffle_unit));
        main_net->Add(bb::LoweringConvolution<T>::Create(MakeDepthwiseLayer<T>({1, 1, 256*w}              ), 3, 3));  // 1x1
        main_net->Add(bb::LoweringConvolution<T>::Create(MakePointwiseLayer<T>({1, 1, 256  }, shuffle_unit), 1, 1));

        main_net->Add(bb::LoweringConvolution<T>::Create(MakePointwiseLayer<T>({1, 1, 512}, shuffle_unit), 1, 1));
        main_net->Add(bb::Shuffle::Create                                     ({1, 1, 512}, shuffle_unit));

        main_net->Add(bb::LoweringConvolution<T>::Create(MakePointwiseLayer<T>({1, 1, 256}, shuffle_unit), 1, 1));
        main_net->Add(bb::Shuffle::Create                                     ({1, 1, 256}, shuffle_unit));

        main_net->Add(bb::LoweringConvolution<T>::Create(MakePointwiseLayer<T>({1, 1, 128}, shuffle_unit), 1, 1));
        main_net->Add(bb::Shuffle::Create                                     ({1, 1, 128}, shuffle_unit));

        main_net->Add(bb::LoweringConvolution<T>::Create(MakePointwiseLayer<T>({1, 1, 64}, shuffle_unit), 1, 1));
        main_net->Add(bb::Shuffle::Create                                     ({1, 1, 64}, shuffle_unit));

        main_net->Add(bb::LoweringConvolution<T>::Create(MakePointwiseLayer<T>({1, 1, 32}, shuffle_unit), 1, 1));
//      main_net->Add(bb::Shuffle::Create                                     ({1, 1, 32}, shuffle_unit);

        main_net->Add(bb::LoweringConvolution<T>::Create(MakeConvLayer<T>({1, 1, 10}), 1, 1));

        main_net->Add(bb::BinaryToReal<T>::Create(train_modulation_size));

        // modulation wrapper
//      auto net = bb::BinaryModulation<float>::Create(main_net, train_modulation_size, test_modulation_size);
        auto net = main_net;

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

        // run fitting
        bb::Runner<float>::create_t runner_create;
        runner_create.name               = net_name;
        runner_create.net                = net;
        runner_create.lossFunc           = bb::LossSoftmaxCrossEntropy<float>::Create();
        runner_create.metricsFunc        = bb::MetricsCategoricalAccuracy<float>::Create();
        runner_create.optimizer          = bb::OptimizerAdam<float>::Create();
        runner_create.file_read          = file_read;       // 前の計算結果があれば読み込んで再開するか
        runner_create.file_write         = true;            // 計算結果をファイルに保存するか
        runner_create.print_progress     = true;            // 途中結果を表示
        runner_create.initial_evaluation = false;//file_read;       // ファイルを読んだ場合は最初に評価しておく 
        auto runner = bb::Runner<float>::Create(runner_create);
        runner->Fitting(td, epoch_size, mini_batch_size);
    }
}


void Cifar10ShuffleNet(int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read)
{
    Cifar10ShuffleNet_<bb::Bit>(epoch_size, mini_batch_size, train_modulation_size, test_modulation_size, binary_mode, file_read);
}


// end of file
