// --------------------------------------------------------------------------
//  BinaryBrain  -- binary network evaluation platform
//   MNIST sample
//
//                                Copyright (C) 2018-2019 by Ryuji Fuchikami
// --------------------------------------------------------------------------


#include <iostream>

#include "bb/Sequential.h"
#include "bb/DenseAffine.h"
#include "bb/BatchNormalization.h"
#include "bb/ReLU.h"
#include "bb/Reduce.h"
#include "bb/DifferentiableLutN.h"
#include "bb/Convolution2d.h"
#include "bb/MaxPooling.h"
#include "bb/BinaryModulation.h"
#include "bb/OptimizerAdam.h"
#include "bb/LossSoftmaxCrossEntropy.h"
#include "bb/MetricsCategoricalAccuracy.h"
#include "bb/Runner.h"
#include "bb/LoadMnist.h"
#include "bb/ExportVerilog.h"



void conv_to_rtl(void)
{
    std::vector< std::shared_ptr< bb::Filter2d<bb::Bit, float> > > layer_list;
    std::vector< std::shared_ptr< bb::DifferentiableLutN<6, bb::Bit> > > depthwise_list;
    std::vector< std::shared_ptr< bb::DifferentiableLutN<6, bb::Bit> > > pointwise_list;
    
    // ネット構築
    auto net = bb::Sequential::Create();

    {
        auto depthwise_lut = bb::DifferentiableLutN<6, bb::Bit>::Create({32, 1, 1});
        depthwise_list.push_back(depthwise_lut);
        auto depthwise_net = bb::Sequential::Create();
        depthwise_net->Add(depthwise_lut);
        auto depthwise_cnv = bb::Convolution2d<bb::Bit>::Create(depthwise_net, 3, 3);
        layer_list.push_back(depthwise_cnv);
        net->Add(depthwise_cnv);
    }

    {
        auto pointwise_lut0 = bb::DifferentiableLutN<6, bb::Bit>::Create({32*6, 1, 1});
        auto pointwise_lut1 = bb::DifferentiableLutN<6, bb::Bit>::Create({32,   1, 1});
        pointwise_list.push_back(pointwise_lut0);
        pointwise_list.push_back(pointwise_lut1);
        auto pointwise_net = bb::Sequential::Create();
        pointwise_net->Add(pointwise_lut0);
        pointwise_net->Add(pointwise_lut1);
        auto pointwise_cnv = bb::Convolution2d<bb::Bit>::Create(pointwise_net, 1, 1);
        layer_list.push_back(pointwise_cnv);
        net->Add(pointwise_cnv);
    }

    for (int i = 0; i < 25; ++i ) {
        {
            auto depthwise_lut = bb::DifferentiableLutN<6, bb::Bit>::Create({8, 1, 32});
            depthwise_list.push_back(depthwise_lut);
            auto depthwise_net = bb::Sequential::Create();
            depthwise_net->Add(depthwise_lut);
            auto depthwise_cnv = bb::Convolution2d<bb::Bit>::Create(depthwise_net, 3, 3);
            layer_list.push_back(depthwise_cnv);
            net->Add(depthwise_cnv);
        }

        {
            auto pointwise_lut0 = bb::DifferentiableLutN<6, bb::Bit>::Create({32*6, 1, 1});
            auto pointwise_lut1 = bb::DifferentiableLutN<6, bb::Bit>::Create({32,   1, 1});
            pointwise_list.push_back(pointwise_lut0);
            pointwise_list.push_back(pointwise_lut1);
            auto pointwise_net = bb::Sequential::Create();
            pointwise_net->Add(pointwise_lut0);
            pointwise_net->Add(pointwise_lut1);
            auto pointwise_cnv = bb::Convolution2d<bb::Bit>::Create(pointwise_net, 1, 1);
            layer_list.push_back(pointwise_cnv);
            net->Add(pointwise_cnv);
        }
    }

    {
        auto depthwise_lut = bb::DifferentiableLutN<6, bb::Bit>::Create({8, 1, 32});
        depthwise_list.push_back(depthwise_lut);
        auto depthwise_net = bb::Sequential::Create();
        depthwise_net->Add(depthwise_lut);
        auto depthwise_cnv = bb::Convolution2d<bb::Bit>::Create(depthwise_net, 3, 3);
        layer_list.push_back(depthwise_cnv);
        net->Add(depthwise_cnv);
    }

    {
        auto pointwise_lut0 = bb::DifferentiableLutN<6, bb::Bit>::Create({64*6, 1, 1});
        auto pointwise_lut1 = bb::DifferentiableLutN<6, bb::Bit>::Create({64,   1, 1});
        pointwise_list.push_back(pointwise_lut0);
        pointwise_list.push_back(pointwise_lut1);
        auto pointwise_net = bb::Sequential::Create();
        pointwise_net->Add(pointwise_lut0);
        pointwise_net->Add(pointwise_lut1);
        auto pointwise_cnv = bb::Convolution2d<bb::Bit>::Create(pointwise_net, 1, 1);
        layer_list.push_back(pointwise_cnv);
        net->Add(pointwise_cnv);
    }

    {
        auto pointwise_lut0 = bb::DifferentiableLutN<6, bb::Bit>::Create({36*6, 1, 1});
        auto pointwise_lut1 = bb::DifferentiableLutN<6, bb::Bit>::Create({36,   1, 1});
        pointwise_list.push_back(pointwise_lut0);
        pointwise_list.push_back(pointwise_lut1);
        auto pointwise_net = bb::Sequential::Create();
        pointwise_net->Add(pointwise_lut0);
        pointwise_net->Add(pointwise_lut1);
        auto pointwise_cnv = bb::Convolution2d<bb::Bit>::Create(pointwise_net, 1, 1);
        layer_list.push_back(pointwise_cnv);
        net->Add(pointwise_cnv);
    }

    {
        auto pointwise_lut0 = bb::DifferentiableLutN<6, bb::Bit>::Create({11*6, 1, 1});
        auto pointwise_lut1 = bb::DifferentiableLutN<6, bb::Bit>::Create({11,   1, 1});
        pointwise_list.push_back(pointwise_lut0);
        pointwise_list.push_back(pointwise_lut1);
        auto pointwise_net = bb::Sequential::Create();
        pointwise_net->Add(pointwise_lut0);
        pointwise_net->Add(pointwise_lut1);
        auto pointwise_cnv = bb::Convolution2d<bb::Bit>::Create(pointwise_net, 1, 1);
        layer_list.push_back(pointwise_cnv);
        net->Add(pointwise_cnv);
    }
    net->SetInputShape({110, 110, 1});

#if 0
    // 読み込み
    for ( int i = 0; i < (int)depthwise_list.size(); ++i ) {
        char finename[64];
        sprintf_s<64>(finename, "mnist_seg/depthwise_cnv%d/sparse_lut.json", i);
        if ( !depthwise_list[i]->LoadJson(finename) ) { printf("read error\n"); }
    }

    for ( int i = 0; i < (int)pointwise_list.size() / 2; ++i ) {
        char finename[64];
        sprintf_s<64>(finename, "mnist_seg/pointwise_cnv%d/sparse_lut_0.json", i);
        if ( !pointwise_list[i*2+0]->LoadJson(finename) ) { printf("read error\n"); }
        sprintf_s<64>(finename, "mnist_seg/pointwise_cnv%d/sparse_lut_1.json", i);
        if ( !pointwise_list[i*2+1]->LoadJson(finename) ) { printf("read error\n"); }
    }
#endif

    net->PrintInfo();

    std::ofstream ofs("mnist_seg.v");
    bb::ExportVerilog_LutCnnLayersAxi4s<bb::Bit>(ofs, "mnist_seg", layer_list);
}



static std::vector< std::vector<float> > make_t(std::vector< std::vector<float> > const &x, std::vector< std::vector<float> > const &t)
{
    std::vector< std::vector<float> > t_img;

    for ( int i = 0; i < (int)x.size(); ++i ) {
        std::vector<float> t_vec(11*28*28, 0);
        for ( int j = 0; j < 10; ++j ) {
            for ( int k = 0; k < 28*28; ++k ) {
                t_vec[j*28*28 + k] = (x[i][k] > 0.5) ? t[i][j] : 0.0f;
            }
        }
        for ( int k = 0; k < 28*28; ++k ) {
            t_vec[10*28*28 + k] = (x[i][k] <= 0.5) ? 1.0f : 0.0f;
        }
        t_img.push_back(t_vec);
    }

    return t_img;
}


static int color_table[11][3] =
{
    {0xe6, 0x00, 0x12},  // 0
    {0x92, 0x07, 0x83},  // 1
    {0x1d, 0x20, 0x88},  // 2
    {0x00, 0x68, 0xb7},  // 3
    {0x00, 0xa0, 0xe9},  // 4
    {0x00, 0x9e, 0x96},  // 5
    {0x00, 0x99, 0x44},  // 6
    {0x8f, 0xc3, 0x1f},  // 7
    {0xff, 0xf1, 0x00},  // 8
    {0xf3, 0x98, 0x00},  // 9
    {0x00, 0x00, 0x00},  // BGC
};


static int argmax_img(std::vector<float> const &img, int pix, int pix_size=56*56, int ch_size=11)
{
    float max_val = 0;
    int   max_c   = 0;
    for ( int c = 0; c < ch_size; c++ ) {
        auto val = img[pix_size*c + pix];
        if ( val > max_val ) {
            max_val = val;
            max_c   = c;
        }
    }
    return max_c;
}


static void write_ppm(std::string filename, std::vector<float> const &img, int w=2*28, int h=2*28)
{
    std::ofstream ofs(filename);
    ofs << "P3\n";
    ofs << w << " " << h << "\n";
    ofs << "255\n";
    for ( int i = 0; i < w*h; ++i ) {
        auto c = argmax_img(img, i, w*h);
        ofs << color_table[c][0] << " " << color_table[c][1] << " " << color_table[c][2] << "\n";
    }
}


static void write_ppm(std::string filename, bb::FrameBuffer const &buf, int frame, int w=2*28, int h=2*28)
{
    write_ppm(filename, buf.GetVector<float>(frame), w , h);
}



static void make_td(std::vector< std::vector<float> > &src_x, std::vector< std::vector<float> > &src_t)
{
    std::vector< std::vector<float> > dst_x;
    std::vector< std::vector<float> > dst_t;

    std::mt19937_64 mt(1);
    std::uniform_int_distribution<int> dist(0, 10);

    for ( int i = 0; i < (int)src_x.size(); i += 4 ) {
        std::vector<float> x_vec(1*56*56);
        std::vector<float> t_vec(11*56*56);
        for ( int yy = 0; yy < 2; ++yy ) {
            for ( int xx = 0; xx < 2; ++xx ) {
                int idx = i + yy*2 + xx;
                for ( int y = 0; y < 28; ++y ) {
                    for ( int x = 0; x < 28; ++x ) {
                        int pix = (yy*28 + y) * 56 + (xx * 28 + x);
                        x_vec[pix] = src_x[idx][y * 28 + x];
                        for ( int c = 0; c < 11; ++c ) {
                            int node = c*56*56 + pix;
                            if ( c < 10 ) {
                                t_vec[node] = (x_vec[pix] > 0.5) ? src_t[idx][c] : 0.0f;
                            }
                            else {
                                t_vec[node] = (dist(mt) == 0 && x_vec[pix] <= 0.5) ? 0.15f : 0.0f;
                            }
                        }
                    }
                }
            }
        }
        dst_x.push_back(x_vec);
        dst_t.push_back(t_vec);
    }

    src_x = dst_x;
    src_t = dst_t;
}


static std::shared_ptr<bb::Model> make_dense_cnv(int ch_size)
{
    auto cnv_net = bb::Sequential::Create();
    cnv_net->Add(bb::BinaryToReal<bb::Bit>::Create());
    cnv_net->Add(bb::DenseAffine<float>::Create(ch_size));
    cnv_net->Add(bb::BatchNormalization<float>::Create());
    cnv_net->Add(bb::Binarize<bb::Bit>::Create());
    return bb::Convolution2d<bb::Bit>::Create(cnv_net, 3, 3, 1, 1, "same");
}

static std::shared_ptr<bb::Model> make_lut_depthwise_cnv(bb::indices_t output_shape, int w=3, int h=3, bool bn=true)
{
    auto cnv_net = bb::Sequential::Create();
    cnv_net->Add(bb::DifferentiableLutN<6, bb::Bit>::Create(output_shape, bn, "depthwise"));
    return bb::Convolution2d<bb::Bit>::Create(cnv_net, w, h, 1, 1, "same");
}

static std::shared_ptr<bb::Model> make_lut_pointwise_cnv(int ch_size, int lut_size=2, bool bn=true)
{
    auto cnv_net = bb::Sequential::Create();
    if ( lut_size >= 3) { cnv_net->Add(bb::DifferentiableLutN<6, bb::Bit>::Create(ch_size*6*6, bn, "random")); }
    if ( lut_size >= 2) { cnv_net->Add(bb::DifferentiableLutN<6, bb::Bit>::Create(ch_size*6,   bn, "random")); }
    if ( lut_size >= 1) { cnv_net->Add(bb::DifferentiableLutN<6, bb::Bit>::Create(ch_size,     bn, "serial")); }
    return bb::Convolution2d<bb::Bit>::Create(cnv_net, 3, 3, 1, 1, "same");
}



void MnistSegmentationDifferentiableLutCnn(int epoch_size, int mini_batch_size, int train_modulation_size, int test_modulation_size, bool binary_mode, bool file_read)
{
//    conv_to_rtl();
//    return;

    std::string net_name = "MnistSegmentationDifferentiableLutCnn";

    // load MNIST data
#ifdef _DEBUG
    auto td = bb::LoadMnist<>::Load(512, 128);
    std::cout << "!!! debug mode !!!" << std::endl;
#else
    auto td = bb::LoadMnist<>::Load();
#endif

#if 0
    td.t_train = make_t(td.x_train, td.t_train);
    td.t_test  = make_t(td.x_test,  td.t_test);
    td.t_shape.resize(3);
    td.t_shape[0] = 28;
    td.t_shape[1] = 28;
    td.t_shape[2] = 11;
#else
    make_td(td.x_train, td.t_train);
    make_td(td.x_test,  td.t_test);
    td.x_shape.resize(3);
    td.x_shape[0] = 28*2;
    td.x_shape[1] = 28*2;
    td.x_shape[2] = 1;
    td.t_shape.resize(3);
    td.t_shape[0] = 28*2;
    td.t_shape[1] = 28*2;
    td.t_shape[2] = 11;
#endif

    write_ppm("t_test0.ppm",  td.t_test[0]);
    write_ppm("t_test1.ppm",  td.t_test[1]);
    write_ppm("t_test2.ppm",  td.t_test[2]);
    write_ppm("t_test3.ppm",  td.t_test[3]);


    // create network
    auto main_net = bb::Sequential::Create();
    main_net->Add(make_lut_depthwise_cnv({36, 1, 1}, 3, 3));
    main_net->Add(make_lut_pointwise_cnv(36, 2, true));
    for ( int i = 0; i < 27; ++i ) {
        main_net->Add(make_lut_depthwise_cnv({8, 1, 36}, 3, 3, true));
        main_net->Add(make_lut_pointwise_cnv(36, 2, true));
    }
    main_net->Add(make_lut_pointwise_cnv(128, 2, true));
    main_net->Add(make_lut_pointwise_cnv(64, 2, true));
    main_net->Add(make_lut_pointwise_cnv(36, 2, true));
    main_net->Add(make_lut_pointwise_cnv(11, 2, false));

    // modulation wrapper
    auto net = bb::Sequential::Create();
    net->Add(bb::BinaryModulation<bb::Bit>::Create(main_net, train_modulation_size, test_modulation_size));
    net->Add(bb::Reduce<>::Create(td.t_shape));
//    auto net = main_net;

    // set input shape
    net->SetInputShape(td.x_shape);

    // set binary mode
    if ( binary_mode ) {
        net->SendCommand("binary true");
    }
    else {
        net->SendCommand("binary false");
    }

//      net->SendCommand("parameter_lock true");

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

    for ( int epoch = 0; epoch < epoch_size; ++epoch ) {
        // run fitting
        bb::Runner<float>::create_t runner_create;
        runner_create.name               = net_name;
        runner_create.net                = net;
        runner_create.lossFunc           = bb::LossSoftmaxCrossEntropy<float>::Create();
        runner_create.metricsFunc        = bb::MetricsCategoricalAccuracy<float>::Create();
        runner_create.optimizer          = bb::OptimizerAdam<float>::Create();
        runner_create.file_read          = (epoch == 0 && file_read);       // 前の計算結果があれば読み込んで再開するか
        runner_create.file_write         = true;            // 計算結果をファイルに保存するか
        runner_create.print_progress     = true;            // 途中結果を表示
        runner_create.initial_evaluation = false;//file_read;       // ファイルを読んだ場合は最初に評価しておく 
        auto runner = bb::Runner<float>::Create(runner_create);
//      runner->Fitting(td, epoch_size, mini_batch_size);
        runner->Fitting(td, 1, mini_batch_size);

        // write pgm
        {
            bb::FrameBuffer x_buf(32, {28*2, 28*2, 1}, BB_TYPE_FP32);
            x_buf.SetVector(td.x_test, 0);
            auto y_buf = net->Forward(x_buf, false);
            write_ppm("seg_0.ppm", y_buf, 0);
            write_ppm("seg_1.ppm", y_buf, 1);
            write_ppm("seg_2.ppm", y_buf, 2);
            write_ppm("seg_3.ppm", y_buf, 3);
            write_ppm("seg_4.ppm", y_buf, 4);
            write_ppm("seg_5.ppm", y_buf, 5);
            write_ppm("seg_6.ppm", y_buf, 6);
            write_ppm("seg_7.ppm", y_buf, 7);
            write_ppm("seg_8.ppm", y_buf, 8);
            write_ppm("seg_9.ppm", y_buf, 9);
            write_ppm("seg_10.ppm", y_buf, 10);
            write_ppm("seg_11.ppm", y_buf, 11);
            write_ppm("seg_12.ppm", y_buf, 12);
            write_ppm("seg_13.ppm", y_buf, 13);
            write_ppm("seg_14.ppm", y_buf, 14);
            write_ppm("seg_15.ppm", y_buf, 15);
            write_ppm("seg_16.ppm", y_buf, 16);
        }
    }
}


// end of file
