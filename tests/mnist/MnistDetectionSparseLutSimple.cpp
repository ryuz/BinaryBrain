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
#include "bb/LossMeanSquaredError.h"
#include "bb/MetricsMeanSquaredError.h"
#include "bb/Runner.h"
#include "bb/LoadMnist.h"
#include "bb/ExportVerilog.h"



// #include "opencv2/opencv.hpp"

#if 0
template <typename T=float, int w=28, int h=28>
void MakeMnistValidationTrainData(
        std::vector< std::vector<T> > const &src_img,
        std::vector< std::vector<T> >       &dst_img,
        std::vector< std::vector<T> >       &dst_t,
        int                                 size,
        std::uint64_t                       seed=1,
        int                                 unit=256,
        int                                 xn=8,
        int                                 yn=8)
{
    std::mt19937_64 mt(seed);

    int array_width  = w*(xn+1);
    int array_height = h*(yn+1);
    std::vector<T>  array_img(array_width*array_height, 0);

    for ( int i = 0; i < size; ++i ) {
        if ( i % unit == 0 ) {
            // 画像作成
            for ( int blk_y = 0; blk_y < xn; ++blk_y ) {
                for ( int blk_x = 0; blk_x < xn; ++blk_x) {
                    int idx = (int)(mt() % src_img.size());
                    for ( int y = 0; y < h; ++y ) {
                        for ( int x = 0; x < w; ++x) {
                            int xx = blk_x * w + x + (w/2);
                            int yy = blk_y * h + y + (h/2);
                            array_img[array_width*yy + xx] = src_img[idx][y*w+x];
                        }
                    }
                }
            }
//            cv::Mat img_cv(array_height, array_height, CV_32F, &array_img[0]);
//            cv::imshow("img", img_cv);
//            cv::waitKey();
        }

        int base_x = (int)(mt() % (w * xn));
        int base_y = (int)(mt() % (h * yn));

        std::vector<T>  img(w*h);
        std::vector<T>  t(1);
        for ( int y = 0; y < h; ++y ) {
            for ( int x = 0; x < w; ++x) {
                int xx = base_x + x;
                int yy = base_y + y;
                img[y*w+x] = array_img[array_width*yy + xx];
            }
        }
        int off_x = base_x % w; 
        int off_y = base_y % h;
        t[0] = 0;
        if ( off_x >= (w/2 - 3) && off_x < (w/2 + 3) && off_y >= (h/2 - 3) && off_y < (h/2 + 3) ) {
            t[0] = (T)1.0;
        }
        else if ( off_x >= (w/2 - 5) && off_x < (w/2 + 5) && off_y >= (h/2 - 5) && off_y < (h/2 + 5) ) {
            t[0] = (T)0.5;
        }

        // 追加
        dst_img.push_back(img);
        dst_t.push_back(t);

//      printf("%d %d %f\n", off_x, off_y, t[0]);
//      cv::Mat img_cv(h, w, CV_32F, &img[0]);
//      cv::imshow("img", img_cv);
//      cv::waitKey();
    }
}

#endif


void MnistDetectionDifferentiableLutSimple(int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read)
{
    std::string net_name = "MnistDetectionDifferentiableLutSimple";

#if 0
  // load MNIST data
#ifdef _DEBUG
    auto td_src = bb::LoadMnist<>::Load(64, 32);
    std::cout << "!!! debug mode !!!" << std::endl;
#else
    auto td_src = bb::LoadMnist<>::Load();
#endif

    /*
    std::vector< std::vector<float> >   dst_img;
    std::vector< std::vector<float> >   dst_t;
    MakeMnistValidationTrainData<>(td.x_train, dst_img, dst_t, 100, 1);
    return;
    */
    bb::TrainData<float> td;
    td.x_shape = bb::indices_t({28, 28, 1});
    td.t_shape = bb::indices_t({1});
    MakeMnistValidationTrainData<>(td_src.x_train, td.x_train, td.t_train, 60000, 1);
    MakeMnistValidationTrainData<>(td_src.x_test,  td.x_test,  td.t_test,  10000, 2);
#endif

  // load MNIST data
#ifdef _DEBUG
    auto td = bb::LoadMnist<>::LoadDetection(64, 32);
    std::cout << "!!! debug mode !!!" << std::endl;
#else
    auto td = bb::LoadMnist<>::LoadDetection();
#endif

    int N = 1;

    auto layer_sl0 = bb::DifferentiableLutN<6, float>::Create(N*6*6*6*6);
    auto layer_sl1 = bb::DifferentiableLutN<6, float>::Create(N*6*6*6);
    auto layer_sl2 = bb::DifferentiableLutN<6, float>::Create(N*6*6);
    auto layer_sl3 = bb::DifferentiableLutN<6, float>::Create(N*6);
    auto layer_sl4 = bb::DifferentiableLutN<6, float>::Create(N*1);

    {
        std::cout << "\n<Training>" << std::endl;

        // main network
        auto main_net = bb::Sequential::Create();
        main_net->Add(layer_sl0);
        main_net->Add(layer_sl1);
        main_net->Add(layer_sl2);
        main_net->Add(layer_sl3);
        main_net->Add(layer_sl4);

        // modulation wrapper
        auto net = bb::Sequential::Create();
        net->Add(bb::BinaryModulation<float>::Create(main_net, train_modulation_size, test_modulation_size));
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
        runner_create.lossFunc           = bb::LossMeanSquaredError<float>::Create();
        runner_create.metricsFunc        = bb::MetricsMeanSquaredError<float>::Create();
        runner_create.optimizer          = bb::OptimizerAdam<float>::Create();
        runner_create.file_read          = file_read;       // 前の計算結果があれば読み込んで再開するか
        runner_create.file_write         = true;            // 計算結果をファイルに保存するか
        runner_create.print_progress     = true;            // 途中結果を表示
        runner_create.initial_evaluation = file_read;       // ファイルを読んだ場合は最初に評価しておく
        auto runner = bb::Runner<float>::Create(runner_create);
        runner->Fitting(td, epoch_size, mini_batch_size);
    }

    {
        std::cout << "\n<Evaluation binary LUT-Network>" << std::endl;

        // LUT-network
        auto layer_bl0 = bb::BinaryLutN<6, bb::Bit>::Create(layer_sl0->GetOutputShape());
        auto layer_bl1 = bb::BinaryLutN<6, bb::Bit>::Create(layer_sl1->GetOutputShape());
        auto layer_bl2 = bb::BinaryLutN<6, bb::Bit>::Create(layer_sl2->GetOutputShape());
        auto layer_bl3 = bb::BinaryLutN<6, bb::Bit>::Create(layer_sl3->GetOutputShape());
        auto layer_bl4 = bb::BinaryLutN<6, bb::Bit>::Create(layer_sl4->GetOutputShape());
        
        auto cnv0_sub = bb::Sequential::Create();
        cnv0_sub->Add(layer_bl0);
        cnv0_sub->Add(layer_bl1);
        cnv0_sub->Add(layer_bl2);
        cnv0_sub->Add(layer_bl3);
        cnv0_sub->Add(layer_bl4);
        auto cnv0 = bb::Convolution2d<bb::Bit>::Create(cnv0_sub, 28, 28);

        auto lut_net = bb::Sequential::Create();
        lut_net->Add(cnv0);

        // evaluation network
        auto eval_net = bb::Sequential::Create();
        eval_net->Add(bb::BinaryModulation<bb::Bit>::Create(lut_net, test_modulation_size));
        eval_net->Add(bb::Reduce<>::Create(td.t_shape));

        // set input shape
        eval_net->SetInputShape(td.x_shape);

        eval_net->PrintInfo();

        // テーブル化して取り込み(SetInputShape後に取り込みが必要)
        std::cout << "parameter copy to binary LUT-Network" << std::endl;
        layer_bl0->ImportLayer(layer_sl0);
        layer_bl1->ImportLayer(layer_sl1);
        layer_bl2->ImportLayer(layer_sl2);
        layer_bl3->ImportLayer(layer_sl3);
        layer_bl4->ImportLayer(layer_sl4);


        // evaluation
        if ( 1 ) {
            std::cout << "test_modulation_size  : " << test_modulation_size  << std::endl;
            bb::Runner<float>::create_t lut_runner_create;
            lut_runner_create.name           = "Lut_" + net_name;
            lut_runner_create.net            = eval_net;
            lut_runner_create.lossFunc       = bb::LossMeanSquaredError<float>::Create();
            lut_runner_create.metricsFunc    = bb::MetricsMeanSquaredError<float>::Create();
            lut_runner_create.optimizer      = bb::OptimizerAdam<float>::Create();
            lut_runner_create.print_progress = true;
            auto lut_runner = bb::Runner<float>::Create(lut_runner_create);
            auto lut_accuracy = lut_runner->Evaluation(td, mini_batch_size);
            std::cout << "lut_accuracy : " << lut_accuracy << std::endl;
        }


        if (1) {
            // Verilog 出力
            std::vector< std::shared_ptr< bb::Filter2d<bb::Bit> > >  vec_cnv0;

            vec_cnv0.push_back(cnv0);

            std::string filename = "verilog/" + net_name + ".v";
            std::ofstream ofs(filename);
            ofs << "`timescale 1ns / 1ps\n\n";
            bb::ExportVerilog_LutCnnLayersAxi4s(ofs, net_name + "Cnv0", vec_cnv0);
            std::cout << "export : " << filename << "\n" << std::endl;
        }

        if (0) {
            // Verilog 出力
            std::string filename = "verilog/" + net_name + ".v";
            std::ofstream ofs(filename);
            ofs << "`timescale 1ns / 1ps\n\n";
            bb::ExportVerilog_LutLayers<>(ofs, net_name, lut_net);
            std::cout << "export : " << filename << "\n" << std::endl;

            // RTL simulation 用データの出力
            bb::WriteTestDataBinTextFile<float>("verilog/mnist_train.txt", "verilog/mnist_test.txt", td);
        }
    }
}


// end of file
