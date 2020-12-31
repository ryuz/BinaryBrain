// --------------------------------------------------------------------------
//  BinaryBrain  -- binary network evaluation platform
//   CIFAR-10 sample
//
//                                Copyright (C) 2018-2019 by Ryuji Fuchikami
// --------------------------------------------------------------------------


#include <iostream>

#include "bb/Sequential.h"
#include "bb/DenseAffine.h"
#include "bb/BatchNormalization.h"
#include "bb/ReLU.h"
#include "bb/LoweringConvolution.h"
#include "bb/MaxPooling.h"
#include "bb/BinaryModulation.h"
#include "bb/OptimizerAdam.h"
#include "bb/LossSoftmaxCrossEntropy.h"
#include "bb/MetricsCategoricalAccuracy.h"
#include "bb/Runner.h"
#include "bb/LoadCifar10.h"


template<typename T=float>
std::vector<T> image_expand_channel(std::vector<T> src_img, bb::index_t ch_size=3, bb::index_t mul=12)
{
    auto img_size = src_img.size() / ch_size;
    std::vector<T> dst_img(ch_size*mul*img_size);
    T   step = (T)1.0 / (mul + 1);
    for ( bb::index_t c = 0; c < ch_size; ++c ) {
        T   lo_th = 0;
        T   hi_th = step;
        for ( int i = 0; i < mul; ++i ) {
            for ( int j = 0; j < img_size; ++j ) {
                T src_val = src_img[c*img_size + j];
                T dst_val = 0;
                if ( src_val >= hi_th ) {
                    dst_val = (T)1.0;
                }
                else if ( src_val >= lo_th ) {
                    dst_val = (T)((src_val - lo_th) / step);
                }
                dst_img[(c*mul+i)*img_size + j] = dst_val;
            }
            lo_th  = hi_th;
            hi_th += step;
        }
    }
    return dst_img;
}

template<typename T=float>
void images_expand_channel(std::vector< std::vector<T> >& src_imgs, bb::index_t ch_size=3, bb::index_t mul=12)
{
    for ( auto& img : src_imgs ) {
        img = image_expand_channel(img, ch_size, mul);
    }
}


void Cifar10DenseCnn(int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read)
{
    std::string net_name = "Cifar10DenseCnn";

  // load cifar-10 data
#ifdef _DEBUG
    auto td = bb::LoadCifar10<>::Load(1);
    std::cout << "!!! debug mode !!!" << std::endl;
#else
    auto td = bb::LoadCifar10<>::Load();
#endif

#if 0
    images_expand_channel<float>(td.x_train);
    images_expand_channel<float>(td.x_test);
    td.x_shape = bb::indices_t({32, 32, 3*12});
#endif

    float momentam = 0.9f;
    if ( binary_mode ) {
        momentam = 0.1f;
    }

    {
        std::cout << "\n<Training>" << std::endl;

        // create network
        auto cnv0_net = bb::Sequential::Create();
        cnv0_net->Add(bb::DenseAffine<>::Create(64));
        cnv0_net->Add(bb::BatchNormalization<>::Create(momentam));
        cnv0_net->Add(bb::ReLU<float>::Create());

        auto cnv1_net = bb::Sequential::Create();
        cnv1_net->Add(bb::DenseAffine<>::Create(64));
        cnv1_net->Add(bb::BatchNormalization<>::Create(momentam));
        cnv1_net->Add(bb::ReLU<float>::Create());

        auto cnv2_net = bb::Sequential::Create();
        cnv2_net->Add(bb::DenseAffine<>::Create(128));
        cnv2_net->Add(bb::BatchNormalization<>::Create(momentam));
        cnv2_net->Add(bb::ReLU<float>::Create());

        auto cnv3_net = bb::Sequential::Create();
        cnv3_net->Add(bb::DenseAffine<>::Create(128));
        cnv3_net->Add(bb::BatchNormalization<>::Create(momentam));
        cnv3_net->Add(bb::ReLU<float>::Create());

        auto main_net = bb::Sequential::Create();
        main_net->Add(bb::LoweringConvolution<>::Create(cnv0_net, 3, 3));   // Conv3x3 x 32
        main_net->Add(bb::LoweringConvolution<>::Create(cnv1_net, 3, 3));   // Conv3x3 x 32
        main_net->Add(bb::MaxPooling<>::Create(2, 2));
        main_net->Add(bb::LoweringConvolution<>::Create(cnv2_net, 3, 3));   // Conv3x3 x 64
        main_net->Add(bb::LoweringConvolution<>::Create(cnv3_net, 3, 3));   // Conv3x3 x 64
        main_net->Add(bb::MaxPooling<>::Create(2, 2));
        main_net->Add(bb::DenseAffine<float>::Create(512));
        main_net->Add(bb::ReLU<float>::Create());
        main_net->Add(bb::DenseAffine<float>::Create(td.t_shape));
        if ( binary_mode ) {
            main_net->Add(bb::BatchNormalization<float>::Create());
            main_net->Add(bb::ReLU<float>::Create());
        }

        // modulation wrapper
        auto net = bb::Sequential::Create();
        if ( binary_mode ) {
//          net->Add(bb::BinaryModulation<float>::Create(main_net, train_modulation_size, test_modulation_size));
            net->Add(bb::RealToBinary<float>::Create(train_modulation_size, 12));
            net->Add(main_net);
            net->Add(bb::BinaryToReal<float>::Create(train_modulation_size));
        }
        else {
            net->Add(main_net);
        }

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


// end of file
