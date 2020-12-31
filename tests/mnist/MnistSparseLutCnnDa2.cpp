// --------------------------------------------------------------------------
//  BinaryBrain  -- binary network evaluation platform
//   MNIST sample
//
//                                Copyright (C) 2018-2019 by Ryuji Fuchikami
// --------------------------------------------------------------------------


#include <iostream>

#include "bb/Sequential.h"
#include "bb/DifferentiableLutN.h"
#include "bb/DifferentiableLutDiscreteN.h"
#include "bb/BinaryLutN.h"
#include "bb/Reduce.h"
#include "bb/BinaryModulation.h"
#include "bb/OptimizerAdam.h"
#include "bb/LossSoftmaxCrossEntropy.h"
#include "bb/MetricsCategoricalAccuracy.h"
#include "bb/Runner.h"
#include "bb/LoadMnist.h"
#include "bb/ExportVerilog.h"

#include "opencv2/opencv.hpp"


template <typename T=float, int width=28, int height=28>
void ImageAugmentation(std::vector<T>& img_src, std::mt19937_64& mt)
{
    cv::Mat img(height, width, cv::DataType<T>::type, &img_src[0]);

    std::uniform_real_distribution<double>  uniform_dist(0.0, 1.0);
//    std::normal_distribution<double>        norm_dist(0.0, 1.0);
    double angle   = uniform_dist(mt) * 10 - 5;
    double scale   = uniform_dist(mt) * 0.5 + 0.7;
    double shift_x = uniform_dist(mt) * 12 - 6;
    double shift_y = uniform_dist(mt) * 12 - 6;

    cv::Mat mat = cv::getRotationMatrix2D(cv::Point2f(width/2, height/2), angle, scale);
    mat.at<double>(0, 2) += shift_x;
    mat.at<double>(1, 2) += shift_y;
    cv::Mat img_dst;
    cv::warpAffine(img, img, mat, img.size());

//  cv::imshow("img", img);
//  cv::waitKey();
}


template <typename T=float, int width=28, int height=28>
cv::Mat MakeMaskImage(void)
{
    cv::Mat mask = cv::Mat::zeros(height, width, cv::DataType<T>::type);
    cv::rectangle(mask, cv::Rect(0, 0, width, height), cv::Scalar(1.0));
    return mask;
}


template <typename T=float, int width=28, int height=28>
bool CheckMask(std::vector<T>& img_src, cv::Mat& mask)
{
    cv::Mat img(height, width, cv::DataType<T>::type, &img_src[0]);
    cv::Mat test = mask.mul(img);
    double maxVal;
    cv::minMaxLoc(test, NULL, &maxVal);
    return maxVal < 0.8;
}


template <typename T=float, int width=28, int height=28>
void DataAugmentation(std::vector<T>& x, std::vector<T>& t, cv::Mat& mask, std::mt19937_64& mt)
{
    if ( mt() % 2 == 1 ) {
        ImageAugmentation<T, width, height>(x, mt);
    }

    /*
    if ( !CheckMask<T, width, height>(x, mask) ) {
        for ( auto& v : t ) { v = 0; }
        *t.rbegin() = 1.0f;
    }
    */

    if ( mt() % 2 == 1 ) {
        for ( auto& v : x ) { v = 1.0f - v; }
    }

    if ( mt() % 100 == 0 ) {
        int idx = (int)(mt() % 100);
        cv::Mat img(height, width, cv::DataType<T>::type, &x[0]);
        cv::imwrite("out/img" + std::to_string(idx) + ".png", img*255.0);
    }
}

static void DataAugmentationProc(bb::TrainData<>& td, std::uint64_t seed, void *user)
{
    std::mt19937_64 mt(seed);
    auto mask = MakeMaskImage<>();

    for ( size_t i = 0; i < td.x_train.size(); ++i ) {
        DataAugmentation<>(td.x_train[i], td.t_train[i], mask, mt);
    }
    for ( size_t i = 0; i < td.x_test.size(); ++i ) {
        DataAugmentation<>(td.x_test[i], td.t_test[i], mask, mt);
    }
}


void MnistDifferentiableLutCnnDa2(int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read)
{
    int N = 7;

    std::string net_name = "MnistDifferentiableLutCnnDa2" + std::to_string(N);

  // load MNIST data
#ifdef _DEBUG
    auto td = bb::LoadMnist<>::Load(64, 32);
    std::cout << "!!! debug mode !!!" << std::endl;
#else
    auto td = bb::LoadMnist<>::Load();
#endif
    
    // create network
    auto layer_cnv0_sl0 = bb::DifferentiableLutN<6, bb::Bit>::Create(192);
    auto layer_cnv0_sl1 = bb::DifferentiableLutN<6, bb::Bit>::Create(32);
    
    auto layer_cnv1_sl0 = bb::DifferentiableLutN<6, bb::Bit>::Create({1, 1, 192}, true, "pointwise");
    auto layer_cnv1_sl1 = bb::DifferentiableLutN<6, bb::Bit>::Create({1, 1,  32}, true, "pointwise");

    auto layer_cnv2_sl0 = bb::DifferentiableLutN<6, bb::Bit>::Create({1, 6,  32}, true, "depthwise");
    auto layer_cnv2_sl1 = bb::DifferentiableLutN<6, bb::Bit>::Create({1, 1,  32}, true, "depthwise");

    auto layer_cnv3_sl0 = bb::DifferentiableLutN<6, bb::Bit>::Create({1, 1, 384}, true, "pointwise");
    auto layer_cnv3_sl1 = bb::DifferentiableLutN<6, bb::Bit>::Create({1, 1,  64}, true, "pointwise");

    auto layer_cnv4_sl0 = bb::DifferentiableLutN<6, bb::Bit>::Create({1, 6,  64}, true, "depthwise");
    auto layer_cnv4_sl1 = bb::DifferentiableLutN<6, bb::Bit>::Create({1, 1,  64}, true, "depthwise");

    auto layer_cnv5_sl0 = bb::DifferentiableLutN<6, bb::Bit>::Create({1, 1, 384}, true, "pointwise");
    auto layer_cnv5_sl1 = bb::DifferentiableLutN<6, bb::Bit>::Create({1, 1,  64}, true, "pointwise");

    auto layer_cnv6_sl0 = bb::DifferentiableLutN<6, bb::Bit>::Create({1, 6,  64}, true, "depthwise");
    auto layer_cnv6_sl1 = bb::DifferentiableLutN<6, bb::Bit>::Create({1, 1,  64}, true, "depthwise");

    auto layer_cnv7_sl0 = bb::DifferentiableLutN<6, bb::Bit>::Create({1, 1, 768}, true, "pointwise");
    auto layer_cnv7_sl1 = bb::DifferentiableLutN<6, bb::Bit>::Create({1, 1, 128}, true, "pointwise");

    auto layer_cnv8_sl0 = bb::DifferentiableLutN<6, bb::Bit>::Create({1, 6, 128}, true, "depthwise");
    auto layer_cnv8_sl1 = bb::DifferentiableLutN<6, bb::Bit>::Create({1, 1, 128}, true, "depthwise");

    auto layer_cnv9_sl0 = bb::DifferentiableLutN<6, bb::Bit>::Create({1, 1, 768}, true, "pointwise");
    auto layer_cnv9_sl1 = bb::DifferentiableLutN<6, bb::Bit>::Create({1, 1, 128}, true, "pointwise");

    auto layer_sl10     = bb::DifferentiableLutN<6, bb::Bit>::Create(N*60);
    auto layer_sl11     = bb::DifferentiableLutN<6, bb::Bit>::Create(N*10);

    {
        std::cout << "\n<Training>" << std::endl;

        // main network
        auto cnv0_sub = bb::Sequential::Create();
        cnv0_sub->Add(layer_cnv0_sl0);
        cnv0_sub->Add(layer_cnv0_sl1);

        auto cnv1_sub = bb::Sequential::Create();
        cnv1_sub->Add(layer_cnv1_sl0);
        cnv1_sub->Add(layer_cnv1_sl1);

        auto cnv2_sub = bb::Sequential::Create();
        cnv2_sub->Add(layer_cnv2_sl0);
        cnv2_sub->Add(layer_cnv2_sl1);

        auto cnv3_sub = bb::Sequential::Create();
        cnv3_sub->Add(layer_cnv3_sl0);
        cnv3_sub->Add(layer_cnv3_sl1);

        auto cnv4_sub = bb::Sequential::Create();
        cnv4_sub->Add(layer_cnv4_sl0);
        cnv4_sub->Add(layer_cnv4_sl1);

        auto cnv5_sub = bb::Sequential::Create();
        cnv5_sub->Add(layer_cnv5_sl0);
        cnv5_sub->Add(layer_cnv5_sl1);

        auto cnv6_sub = bb::Sequential::Create();
        cnv6_sub->Add(layer_cnv6_sl0);
        cnv6_sub->Add(layer_cnv6_sl1);
        
        auto cnv7_sub = bb::Sequential::Create();
        cnv7_sub->Add(layer_cnv7_sl0);
        cnv7_sub->Add(layer_cnv7_sl1);

        auto cnv8_sub = bb::Sequential::Create();
        cnv8_sub->Add(layer_cnv8_sl0);
        cnv8_sub->Add(layer_cnv8_sl1);

        auto cnv9_sub = bb::Sequential::Create();
        cnv9_sub->Add(layer_cnv9_sl0);
        cnv9_sub->Add(layer_cnv9_sl1);

        auto main_net = bb::Sequential::Create();
        main_net->Add(bb::Convolution2d<bb::Bit>::Create(cnv0_sub, 3, 3));
        main_net->Add(bb::Convolution2d<bb::Bit>::Create(cnv1_sub, 1, 1));
        main_net->Add(bb::Convolution2d<bb::Bit>::Create(cnv2_sub, 3, 3));
        main_net->Add(bb::MaxPooling<bb::Bit>::Create(2, 2));
        main_net->Add(bb::Convolution2d<bb::Bit>::Create(cnv3_sub, 1, 1));
        main_net->Add(bb::Convolution2d<bb::Bit>::Create(cnv4_sub, 3, 3));
        main_net->Add(bb::Convolution2d<bb::Bit>::Create(cnv5_sub, 1, 1));
        main_net->Add(bb::Convolution2d<bb::Bit>::Create(cnv6_sub, 3, 3));
        main_net->Add(bb::MaxPooling<bb::Bit>::Create(2, 2));
        main_net->Add(bb::Convolution2d<bb::Bit>::Create(cnv7_sub, 1, 1));
        main_net->Add(bb::Convolution2d<bb::Bit>::Create(cnv8_sub, 3, 3));
        main_net->Add(bb::Convolution2d<bb::Bit>::Create(cnv9_sub, 1, 1));
        main_net->Add(layer_sl10);
        main_net->Add(layer_sl11);

        // modulation wrapper
        auto net = bb::Sequential::Create();
        net->Add(bb::BinaryModulation<bb::Bit>::Create(main_net, train_modulation_size, test_modulation_size));
        net->Add(bb::Reduce<float>::Create(td.t_shape));

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
        std::cout << "-----------------------------------" << std::endl;

        // fitting
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
        runner_create.data_augmentation_proc = DataAugmentationProc;
        auto runner = bb::Runner<float>::Create(runner_create);
        runner->Fitting(td, epoch_size, mini_batch_size);
    }

#if 0
    {
        std::cout << "\n<Evaluation binary LUT-Network>" << std::endl;

        // LUT-network
        auto layer_cnv0_bl0 = bb::BinaryLutN<6, bb::Bit>::Create(layer_cnv0_sl0->GetOutputShape());
        auto layer_cnv0_bl1 = bb::BinaryLutN<6, bb::Bit>::Create(layer_cnv0_sl1->GetOutputShape());
        auto layer_cnv1_bl0 = bb::BinaryLutN<6, bb::Bit>::Create(layer_cnv1_sl0->GetOutputShape());
        auto layer_cnv1_bl1 = bb::BinaryLutN<6, bb::Bit>::Create(layer_cnv1_sl1->GetOutputShape());
        auto layer_cnv2_bl0 = bb::BinaryLutN<6, bb::Bit>::Create(layer_cnv2_sl0->GetOutputShape());
        auto layer_cnv2_bl1 = bb::BinaryLutN<6, bb::Bit>::Create(layer_cnv2_sl1->GetOutputShape());
        auto layer_cnv3_bl0 = bb::BinaryLutN<6, bb::Bit>::Create(layer_cnv3_sl0->GetOutputShape());
        auto layer_cnv3_bl1 = bb::BinaryLutN<6, bb::Bit>::Create(layer_cnv3_sl1->GetOutputShape());
        auto layer_bl4      = bb::BinaryLutN<6, bb::Bit>::Create(layer_sl4->GetOutputShape());
        auto layer_bl5      = bb::BinaryLutN<6, bb::Bit>::Create(layer_sl5->GetOutputShape());
        auto layer_bl6      = bb::BinaryLutN<6, bb::Bit>::Create(layer_sl6->GetOutputShape());
//        auto layer_bl7      = bb::BinaryLutN<6, bb::Bit>::Create(layer_sl7->GetOutputShape());

        auto cnv0_sub = bb::Sequential::Create();
        cnv0_sub->Add(layer_cnv0_bl0);
        cnv0_sub->Add(layer_cnv0_bl1);

        auto cnv1_sub = bb::Sequential::Create();
        cnv1_sub->Add(layer_cnv1_bl0);
        cnv1_sub->Add(layer_cnv1_bl1);

        auto cnv2_sub = bb::Sequential::Create();
        cnv2_sub->Add(layer_cnv2_bl0);
        cnv2_sub->Add(layer_cnv2_bl1);

        auto cnv3_sub = bb::Sequential::Create();
        cnv3_sub->Add(layer_cnv3_bl0);
        cnv3_sub->Add(layer_cnv3_bl1);

        auto cnv4_sub = bb::Sequential::Create();
        cnv4_sub->Add(layer_bl4);
        cnv4_sub->Add(layer_bl5);
        cnv4_sub->Add(layer_bl6);
//        cnv4_sub->Add(layer_bl7);

        auto cnv0 = bb::Convolution2d<bb::Bit>::Create(cnv0_sub, 3, 3);
        auto cnv1 = bb::Convolution2d<bb::Bit>::Create(cnv1_sub, 3, 3);
        auto pol0 = bb::MaxPooling<bb::Bit>::Create(2, 2);

        auto cnv2 = bb::Convolution2d<bb::Bit>::Create(cnv2_sub, 3, 3);
        auto cnv3 = bb::Convolution2d<bb::Bit>::Create(cnv3_sub, 3, 3);
        auto pol1 = bb::MaxPooling<bb::Bit>::Create(2, 2);

        auto cnv4 = bb::Convolution2d<bb::Bit>::Create(cnv4_sub, 4, 4);

        auto lut_net = bb::Sequential::Create();
        lut_net->Add(cnv0);
        lut_net->Add(cnv1);
        lut_net->Add(pol0);
        lut_net->Add(cnv2);
        lut_net->Add(cnv3);
        lut_net->Add(pol1);
        lut_net->Add(cnv4);

        // evaluation network
        auto eval_net = bb::Sequential::Create();
        eval_net->Add(bb::BinaryModulation<bb::Bit>::Create(lut_net, test_modulation_size));
        eval_net->Add(bb::Reduce<>::Create(td.t_shape));

        // set input shape
        eval_net->SetInputShape(td.x_shape);


        // テーブル化して取り込み(現状まだSetInputShape後の取り込みが必要)
        std::cout << "parameter copy to binary LUT-Network" << std::endl;
        layer_cnv0_bl0->ImportLayer(layer_cnv0_sl0);
        layer_cnv0_bl1->ImportLayer(layer_cnv0_sl1);
        layer_cnv1_bl0->ImportLayer(layer_cnv1_sl0);
        layer_cnv1_bl1->ImportLayer(layer_cnv1_sl1);
        layer_cnv2_bl0->ImportLayer(layer_cnv2_sl0);
        layer_cnv2_bl1->ImportLayer(layer_cnv2_sl1);
        layer_cnv3_bl0->ImportLayer(layer_cnv3_sl0);
        layer_cnv3_bl1->ImportLayer(layer_cnv3_sl1);
        layer_bl4     ->ImportLayer(layer_sl4);
        layer_bl5     ->ImportLayer(layer_sl5);
        layer_bl6     ->ImportLayer(layer_sl6);
//        layer_bl7     ->ImportLayer(layer_sl7);

        // 評価
        if ( 1 ) {
            std::cout << "test_modulation_size  : " << test_modulation_size  << std::endl;
            bb::Runner<float>::create_t lut_runner_create;
            lut_runner_create.name        = "Lut_" + net_name;
            lut_runner_create.net         = eval_net;
            lut_runner_create.lossFunc    = bb::LossSoftmaxCrossEntropy<float>::Create();
            lut_runner_create.metricsFunc = bb::MetricsCategoricalAccuracy<float>::Create();
            lut_runner_create.optimizer   = bb::OptimizerAdam<float>::Create();
            lut_runner_create.initial_evaluation = false;
            lut_runner_create.print_progress = true;    // 途中結果を出力
            auto lut_runner = bb::Runner<float>::Create(lut_runner_create);
            auto lut_accuracy = lut_runner->Evaluation(td, mini_batch_size);
            std::cout << "lut_accuracy : " << lut_accuracy << std::endl;
        }

        {
            // Verilog 出力
            std::vector< std::shared_ptr< bb::Filter2d<bb::Bit> > >  vec_cnv0;
            std::vector< std::shared_ptr< bb::Filter2d<bb::Bit> > >  vec_cnv1;
            std::vector< std::shared_ptr< bb::Filter2d<bb::Bit> > >  vec_cnv2;

            vec_cnv0.push_back(cnv0);
            vec_cnv0.push_back(cnv1);
            vec_cnv0.push_back(pol0);
            vec_cnv1.push_back(cnv2);
            vec_cnv1.push_back(cnv3);
            vec_cnv1.push_back(pol1);
            vec_cnv2.push_back(cnv4);

            net_name = "MnistDifferentiableLutCnn";

            std::string filename = "verilog/" + net_name + ".v";
            std::ofstream ofs(filename);
            ofs << "`timescale 1ns / 1ps\n\n";
            bb::ExportVerilog_LutCnnLayersAxi4s(ofs, net_name + "Cnv0", vec_cnv0);
            bb::ExportVerilog_LutCnnLayersAxi4s(ofs, net_name + "Cnv1", vec_cnv1);
            bb::ExportVerilog_LutCnnLayersAxi4s(ofs, net_name + "Cnv2", vec_cnv2);
            std::cout << "export : " << filename << "\n" << std::endl;
            
            // write test image
            bb::WriteTestDataImage<float>("verilog/mnist_test_160x120.ppm", 160, 120, td);
            bb::WriteTestDataImage<float>("verilog/mnist_test_640x480.ppm", 640, 480, td);
        }
    }
#endif
}


// end of file
