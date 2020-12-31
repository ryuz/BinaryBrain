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
#include "bb/DifferentiableLutN.h"
#include "bb/BatchNormalization.h"
#include "bb/ReLU.h"
#include "bb/Convolution2d.h"
#include "bb/MaxPooling.h"
#include "bb/BinaryModulation.h"
#include "bb/OptimizerAdam.h"
#include "bb/LossSoftmaxCrossEntropy.h"
#include "bb/MetricsCategoricalAccuracy.h"
#include "bb/Runner.h"
#include "bb/LoadMnist.h"


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
std::shared_ptr<bb::Model> MakeDepthwiseLayer(bb::indices_t shape)
{
#if 0
    auto net = bb::Sequential::Create();
    net->Add(bb::DepthwiseDenseAffine<float>::Create(shape));
    net->Add(bb::BatchNormalization<float>::Create());
    net->Add(bb::ReLU<T>::Create());
    return net;
#else
    return bb::DifferentiableLutN<6, T>::Create(shape, true, "depthwise");
#endif
}


template <typename T>
void MnistMobileNet_(int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read)
{
    std::string net_name = "MnistMobileNet";
    if ( binary_mode ) {
        net_name += "Bit";
    }

    // load MNIST data
#ifdef _DEBUG
    auto td = bb::LoadMnist<>::Load(512, 128);
    std::cout << "!!! debug mode !!!" << std::endl;
#else
    auto td = bb::LoadMnist<>::Load();
#endif

    {
        std::cout << "\n<Training>" << std::endl;

        // create network
        int w=4;

        auto main_net = bb::Sequential::Create();

        main_net->Add(bb::Convolution2d<T>::Create(MakeConvLayer<T>     ({1, 1,  32*w}), 1, 1));
        main_net->Add(bb::Convolution2d<T>::Create(MakeDepthwiseLayer<T>({1, 1,  32*w}), 3, 3));  // 26x26
        main_net->Add(bb::Convolution2d<T>::Create(MakeConvLayer<T>     ({1, 1,  32  }), 1, 1));

        main_net->Add(bb::Convolution2d<T>::Create(MakeConvLayer<T>     ({1, 1,  32*w}), 1, 1));
        main_net->Add(bb::Convolution2d<T>::Create(MakeDepthwiseLayer<T>({1, 1,  32*w}), 3, 3));  // 24x24
        main_net->Add(bb::Convolution2d<T>::Create(MakeConvLayer<T>     ({1, 1,  32  }), 1, 1));

        main_net->Add(bb::MaxPooling<T>::Create(2, 2));                                                 // 12x12

        main_net->Add(bb::Convolution2d<T>::Create(MakeConvLayer<T>     ({1, 1,  64*w}), 1, 1));
        main_net->Add(bb::Convolution2d<T>::Create(MakeDepthwiseLayer<T>({1, 1,  64*w}), 3, 3));  // 10x10
        main_net->Add(bb::Convolution2d<T>::Create(MakeConvLayer<T>     ({1, 1,  64  }), 1, 1));

        main_net->Add(bb::Convolution2d<T>::Create(MakeConvLayer<T>     ({1, 1,  64*w}), 1, 1));
        main_net->Add(bb::Convolution2d<T>::Create(MakeDepthwiseLayer<T>({1, 1,  64*w}), 3, 3));  // 8x8
        main_net->Add(bb::Convolution2d<T>::Create(MakeConvLayer<T>     ({1, 1,  64  }), 1, 1));

        main_net->Add(bb::MaxPooling<T>::Create(2, 2));                                                 // 4x4

        main_net->Add(bb::Convolution2d<T>::Create(MakeConvLayer<T>     ({1, 1,  128*w}), 1, 1));
        main_net->Add(bb::Convolution2d<T>::Create(MakeDepthwiseLayer<T>({1, 1,  128*w}), 3, 3));  // 2x2
        main_net->Add(bb::Convolution2d<T>::Create(MakeConvLayer<T>     ({1, 1,  128  }), 1, 1));

        main_net->Add(bb::Convolution2d<T>::Create(MakeConvLayer<T>     ({1, 1,  128*w}), 1, 1));
        main_net->Add(bb::Convolution2d<T>::Create(MakeDepthwiseLayer<T>({1, 1,  128*w}), 2, 2));  // 1x1
        main_net->Add(bb::Convolution2d<T>::Create(MakeConvLayer<T>     ({1, 1,  128  }), 1, 1));

        main_net->Add(bb::Convolution2d<T>::Create(MakeConvLayer<T>     ({1, 1,  256  }), 1, 1));

        main_net->Add(bb::DenseAffine<>::Create(td.t_shape));
        if ( binary_mode ) {
            main_net->Add(bb::BatchNormalization<>::Create());
            main_net->Add(bb::ReLU<T>::Create());
        }

        // modulation wrapper
        auto net = bb::BinaryModulation<T>::Create(main_net, train_modulation_size, test_modulation_size, 8);

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
        runner_create.initial_evaluation = file_read;       // ファイルを読んだ場合は最初に評価しておく 
        auto runner = bb::Runner<float>::Create(runner_create);
        runner->Fitting(td, epoch_size, mini_batch_size);
    }
}


void MnistMobileNet(int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read)
{
    if ( binary_mode ) {
        MnistMobileNet_<bb::Bit>(epoch_size, mini_batch_size, train_modulation_size, test_modulation_size, binary_mode, file_read);
    }
    else {
        MnistMobileNet_<float>(epoch_size, mini_batch_size, train_modulation_size, test_modulation_size, binary_mode, file_read);
    }
}


// end of file
