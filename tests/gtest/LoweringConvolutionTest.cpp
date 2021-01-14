#include <stdio.h>
#include <iostream>
#include "gtest/gtest.h"

#include "bb/DataType.h"
#include "bb/NormalDistributionGenerator.h"
#include "bb/UniformDistributionGenerator.h"
#include "bb/Convolution2d.h"
#include "bb/DenseAffine.h"
#include "bb/BinaryLutN.h"
#include "bb/Sequential.h"


TEST(Convolution2dTest, testConvolution2d)
{
    auto sub_affine = bb::DenseAffine<float>::Create(1);
//    1 * 2 * 2

    auto cnv = bb::Convolution2d<>::Create(sub_affine, 2, 2);

    bb::FrameBuffer x_buf(1, {1, 3, 3}, BB_TYPE_FP32);

    cnv->SetInputShape({1, 3, 3});

//  EXPECT_EQ(9, cnv.GetInputNodeSize());
//  EXPECT_EQ(4, cnv.GetOutputNodeSize());

    x_buf.SetFP32(0, 3 * 0 + 0, 0.1f);
    x_buf.SetFP32(0, 3 * 0 + 1, 0.2f);
    x_buf.SetFP32(0, 3 * 0 + 2, 0.3f);
    x_buf.SetFP32(0, 3 * 1 + 0, 0.4f);
    x_buf.SetFP32(0, 3 * 1 + 1, 0.5f);
    x_buf.SetFP32(0, 3 * 1 + 2, 0.6f);
    x_buf.SetFP32(0, 3 * 2 + 0, 0.7f);
    x_buf.SetFP32(0, 3 * 2 + 1, 0.8f);
    x_buf.SetFP32(0, 3 * 2 + 2, 0.9f);

    EXPECT_EQ(0.1f, x_buf.GetFP32(0, { 0, 0, 0 }));
    EXPECT_EQ(0.2f, x_buf.GetFP32(0, { 0, 0, 1 }));
    EXPECT_EQ(0.3f, x_buf.GetFP32(0, { 0, 0, 2 }));
    EXPECT_EQ(0.4f, x_buf.GetFP32(0, { 0, 1, 0 }));
    EXPECT_EQ(0.5f, x_buf.GetFP32(0, { 0, 1, 1 }));
    EXPECT_EQ(0.6f, x_buf.GetFP32(0, { 0, 1, 2 }));
    EXPECT_EQ(0.7f, x_buf.GetFP32(0, { 0, 2, 0 }));
    EXPECT_EQ(0.8f, x_buf.GetFP32(0, { 0, 2, 1 }));
    EXPECT_EQ(0.9f, x_buf.GetFP32(0, { 0, 2, 2 }));

//  cnv.W(0, 0, 0, 0) = 0.1f;
//  cnv.W(0, 0, 0, 1) = 0.2f;
//  cnv.W(0, 0, 1, 0) = 0.3f;
//  cnv.W(0, 0, 1, 1) = 0.4f;

//  cnv.b(0) = 0.321f;

    {
        auto W = sub_affine->lock_W();
        auto b = sub_affine->lock_b();
        W(0, 0) = 0.1f;
        W(0, 1) = 0.2f;
        W(0, 2) = 0.3f;
        W(0, 3) = 0.4f;
        b(0) = 0.321f;
    }


    auto y_buf = cnv->Forward(x_buf);


    float exp00 = 0.321f
        + 0.1f * 0.1f
        + 0.2f * 0.2f
        + 0.4f * 0.3f
        + 0.5f * 0.4f;

    float exp01 = 0.321f
        + 0.2f * 0.1f
        + 0.3f * 0.2f
        + 0.5f * 0.3f
        + 0.6f * 0.4f;

    float exp10 = 0.321f
        + 0.4f * 0.1f
        + 0.5f * 0.2f
        + 0.7f * 0.3f
        + 0.8f * 0.4f;

    float exp11 = 0.321f
        + 0.5f * 0.1f
        + 0.6f * 0.2f
        + 0.8f * 0.3f
        + 0.9f * 0.4f;
    
//  std::cout << exp00 << std::endl;
//  std::cout << exp01 << std::endl;
//  std::cout << exp10 << std::endl;
//  std::cout << exp11 << std::endl;

    EXPECT_NEAR(exp00, y_buf.GetFP32(0, 0), 0.000001);
    EXPECT_NEAR(exp01, y_buf.GetFP32(0, 1), 0.000001);
    EXPECT_NEAR(exp10, y_buf.GetFP32(0, 2), 0.000001);
    EXPECT_NEAR(exp11, y_buf.GetFP32(0, 3), 0.000001);
}




template<typename FT=float, typename BT=float>
void testConvolution2d_cmpare(
        std::shared_ptr<bb::Model> cnv_layer,
        int loop_num,
        int frame_size,
        int y_c = 32,
        int x_c = 3,
        int x_h = 28,
        int x_w = 28,
        int f_h = 3,
        int f_w = 3)
{
    int y_h = x_h - f_h + 1;
    int y_w = x_w - f_w + 1;

    auto layer_cpu = bb::Convolution2d<FT, BT>::Create(cnv_layer, f_h, f_w);
    auto layer_gpu = bb::Convolution2d<FT, BT>::Create(cnv_layer, f_h, f_w);

    bb::FrameBuffer x_cpu(frame_size, {x_c, x_h, x_w}, bb::DataType<FT>::type, true);
    bb::FrameBuffer x_gpu(frame_size, {x_c, x_h, x_w}, bb::DataType<FT>::type);
    
    layer_cpu->SetInputShape(x_cpu.GetShape());
    layer_gpu->SetInputShape(x_gpu.GetShape());

    layer_cpu->SendCommand("host_only true");

    auto valgen = bb::UniformDistributionGenerator<FT>::Create(0, 1, 1);

    for ( int loop = 0; loop < loop_num; ++ loop ) 
    {
        for ( int frame = 0; frame < frame_size; ++frame) {
            for ( int node = 0; node < x_cpu.GetNodeSize(); ++node ) {
                x_cpu.SetFP32(frame, node, valgen->GetValue());
                x_gpu.SetFP32(frame, node, x_cpu.GetFP32(frame, node));
            }
        }

        auto y_cpu = layer_cpu->Forward(x_cpu);
        auto y_gpu = layer_gpu->Forward(x_gpu);

        // 壊れていないことを一応確認
        for ( int frame = 0; frame < frame_size; ++frame) {
            for ( int node = 0; node < x_cpu.GetNodeSize(); ++node ) {
                auto val_cpu = x_cpu.GetFP32(frame, node);
                auto val_gpu = x_gpu.GetFP32(frame, node);
                EXPECT_EQ(val_cpu, val_gpu);
            }
        }

        // 結果比較
        for ( int frame = 0; frame < frame_size; ++frame) {
            for ( int node = 0; node < y_cpu.GetNodeSize(); ++node ) {
                auto val_cpu = y_cpu.GetFP32(frame, node);
                auto val_gpu = y_gpu.GetFP32(frame, node);
                EXPECT_NEAR(val_cpu, val_gpu, 0.0001f);
            }
        }

        if ( bb::DataType<FT>::type != BB_TYPE_BIT ) {
            // backward
            bb::FrameBuffer dy_cpu(frame_size, {y_w, y_h, y_c}, BB_TYPE_FP32, true);
            bb::FrameBuffer dy_gpu(frame_size, {y_w, y_h, y_c}, BB_TYPE_FP32);
            for ( int frame = 0; frame < frame_size; ++frame) {
                for ( int node = 0; node < dy_cpu.GetNodeSize(); ++node ) {
                    dy_cpu.SetFP32(frame, node, valgen->GetValue());
                    dy_gpu.SetFP32(frame, node, dy_cpu.GetFP32(frame, node));
                }
            }

            auto dx_cpu = layer_cpu->Backward(dy_cpu);
            auto dx_gpu = layer_gpu->Backward(dy_gpu);

            for ( int frame = 0; frame < frame_size; ++frame) {
                for ( int node = 0; node < dx_cpu.GetNodeSize(); ++node ) {
                    auto val_cpu = dx_cpu.GetFP32(frame, node);
                    auto val_gpu = dx_gpu.GetFP32(frame, node);
                    EXPECT_NEAR(val_cpu, val_gpu, 0.0001f);
                }
            }
        }
    }
}



TEST(Convolution2dTest, testConvolution2d_cmp_float)
{
//  testConvolution2d_cmpare<float, float>(bb::DenseAffine<float>::Create(32), 2, 16, 32);
}

TEST(Convolution2dTest, testConvolution2d_cmp_bit)
{
//    testConvolution2d_cmpare<bb::Bit, float>(bb::BinaryLutN<6, bb::Bit>::Create(32), 2, 16, 32);
//    testConvolution2d_cmpare<bb::Bit, float>(bb::BinaryLutN<6, bb::Bit>::Create(10), 4, 16*15, 10);
    testConvolution2d_cmpare<bb::Bit, float>(bb::BinaryLutN<6, bb::Bit>::Create(32), 2, 8, 32);
    testConvolution2d_cmpare<bb::Bit, float>(bb::BinaryLutN<6, bb::Bit>::Create(10), 2, 8, 10);
}


#if 0

TEST(NeuralNetConvolution2dTest, testNeuralNetConvolution2d2)
{
    bb::NeuralNetDenseAffine<>  sub_affine(3 * 3 * 3, 2);
    bb::NeuralNetGroup<>        sub_net;
    sub_net.AddLayer(&sub_affine);
    bb::NeuralNetConvolution2d<>  cnv(&sub_net, 3, 4, 5, 2, 3, 3);

//  bb::NeuralNetConvolution<> cnv(3, 4, 5, 2, 3, 3);
    cnv.SetBatchSize(2);

    testSetupLayerBuffer(cnv);
    auto in_val = cnv.GetInputSignalBuffer();
    auto out_val = cnv.GetOutputSignalBuffer();

    in_val.SetDimensions({ 5, 4, 3 });
    int index = 0;
    for (bb::INDEX f = 0; f < 2; ++f) {
        for (bb::INDEX c = 0; c < 3; ++c) {
            for (bb::INDEX y = 0; y < 4; ++y) {
                for (bb::INDEX x = 0; x < 5; ++x) {
                    in_val.SetReal(f, { x, y, c }, (float)(index++));
                }
            }
        }
    }
    EXPECT_EQ(120, index);

    EXPECT_EQ(0, in_val.GetReal(0, { 0, 0, 0 }));
    EXPECT_EQ(23, in_val.GetReal(0, { 3, 0, 1 }));
    EXPECT_EQ(71, in_val.GetReal(1, { 1, 2, 0 }));
    EXPECT_EQ(116, in_val.GetReal(1, { 1, 3, 2 }));
    EXPECT_EQ(119, in_val.GetReal(1, { 4, 3, 2 }));


    index = 10;
//  for (size_t n = 0; n < 2; ++n) {
//      for (size_t c = 0; c < 3; ++c) {
//          for (size_t y = 0; y < 3; ++y) {
//              for (size_t x = 0; x < 3; ++x) {
//                  cnv.W(n, c, y, x) = (float)(index++);
//              }
//          }
//      }
//  }
    for (size_t n = 0; n < 2; ++n) {
        for (size_t i = 0; i < 3 * 3 * 3; ++i) {
            sub_affine.W(n, i) = (float)(index++);
        }
    }

    EXPECT_EQ(64, index);

    index = 100;
    for (size_t n = 0; n < 2; ++n) {
        sub_affine.b(n) = (float)(index++);
    }

    cnv.Forward();
    
    EXPECT_EQ(19774, out_val.GetReal(0, 0));
    EXPECT_EQ(20395, out_val.GetReal(0, 1));
    EXPECT_EQ(21016, out_val.GetReal(0, 2));
    EXPECT_EQ(22879, out_val.GetReal(0, 3));
    EXPECT_EQ(23500, out_val.GetReal(0, 4));
    EXPECT_EQ(24121, out_val.GetReal(0, 5));
    EXPECT_EQ(38729, out_val.GetReal(0, 6));
    EXPECT_EQ(40079, out_val.GetReal(0, 7));
    EXPECT_EQ(41429, out_val.GetReal(0, 8));
    EXPECT_EQ(45479, out_val.GetReal(0, 9));
    EXPECT_EQ(46829, out_val.GetReal(0, 10));
    EXPECT_EQ(48179, out_val.GetReal(0, 11));
    EXPECT_EQ(57034, out_val.GetReal(1, 0));
    EXPECT_EQ(57655, out_val.GetReal(1, 1));
    EXPECT_EQ(58276, out_val.GetReal(1, 2));
    EXPECT_EQ(60139, out_val.GetReal(1, 3));
    EXPECT_EQ(60760, out_val.GetReal(1, 4));
    EXPECT_EQ(61381, out_val.GetReal(1, 5));
    EXPECT_EQ(119729, out_val.GetReal(1, 6));
    EXPECT_EQ(121079, out_val.GetReal(1, 7));
    EXPECT_EQ(122429, out_val.GetReal(1, 8));
    EXPECT_EQ(126479, out_val.GetReal(1, 9));
    EXPECT_EQ(127829, out_val.GetReal(1, 10));
    EXPECT_EQ(129179, out_val.GetReal(1, 11));
    

    // backward

    auto in_err = cnv.GetInputErrorBuffer();
    auto out_err = cnv.GetOutputErrorBuffer();

    in_err.SetDimensions({ 5, 4, 3 });
    out_err.SetDimensions({ 3, 2, 2 });

    index = 30;
    for (bb::INDEX f = 0; f < 2; ++f) {
        for (bb::INDEX c = 0; c < 2; ++c) {
            for (bb::INDEX y = 0; y < 2; ++y) {
                for (bb::INDEX x = 0; x < 3; ++x) {
                    out_err.SetReal(f, { x, y, c }, (float)(index++));
                }
            }
        }
    }
    cnv.Backward();

#if 0
    for (size_t n = 0; n < 2; ++n) {
        for (size_t c = 0; c < 3; ++c) {
            for (size_t y = 0; y < 3; ++y) {
                for (size_t x = 0; x < 3; ++x) {
                    std::cout << cnv.dW(n, c, y, x) << ", ";
                }
                std::cout << std::endl;
            }
        }
    }
#endif

    /*
    array([[[[17690., 18152., 18614.],
         [20000., 20462., 20924.],
         [22310., 22772., 23234.]],

        [[26930., 27392., 27854.],
         [29240., 29702., 30164.],
         [31550., 32012., 32474.]],

        [[36170., 36632., 37094.],
         [38480., 38942., 39404.],
         [40790., 41252., 41714.]]],


       [[[20102., 20636., 21170.],
         [22772., 23306., 23840.],
         [25442., 25976., 26510.]],

        [[30782., 31316., 31850.],
         [33452., 33986., 34520.],
         [36122., 36656., 37190.]],

        [[41462., 41996., 42530.],
         [44132., 44666., 45200.],
         [46802., 47336., 47870.]]]])
     */
    
//  EXPECT_EQ(17690, sub_affine.dW(0, 0, 0, 0));
//  EXPECT_EQ(29702, sub_affine.dW(0, 1, 1, 1));
//  EXPECT_EQ(26510, sub_affine.dW(1, 0, 2, 2));
//  EXPECT_EQ(33452, sub_affine.dW(1, 1, 1, 0));
//  EXPECT_EQ(47870, sub_affine.dW(1, 2, 2, 2));

    EXPECT_EQ(462, sub_affine.db(0));
    EXPECT_EQ(534, sub_affine.db(1));

#if 0
    for (size_t f = 0; f < 2; ++f) {
        for (size_t c = 0; c < 3; ++c) {
            for (size_t y = 0; y < 4; ++y) {
                for (size_t x = 0; x < 5; ++x) {
                    std::cout << in_err.GetReal(f, { x, y, c }) << ", ";
                }
                std::cout << std::endl;
            }
        }
    }
#endif

    /*
    array([[[[1632., 3377., 5237., 3611., 1866.],
        [3603., 7444., 11527., 7936., 4095.],
        [4017., 8284., 12805., 8800., 4533.],
        [2205., 4541., 7010., 4811., 2475.]],

        [[2226., 4583., 7073., 4853., 2496.],
        [4845., 9964., 15361., 10528., 5409.],
        [5259., 10804., 16639., 11392., 5847.],
        [2853., 5855., 9008., 6161., 3159.]],

        [[2820., 5789., 8909., 6095., 3126.],
        [6087., 12484., 19195., 13120., 6723.],
        [6501., 13324., 20473., 13984., 7161.],
        [3501., 7169., 11006., 7511., 3843.]]],


        [[[2196., 4529., 7001., 4811., 2478.],
        [4803., 9892., 15271., 10480., 5391.],
        [5361., 11020., 16981., 11632., 5973.],
        [2913., 5981., 9206., 6299., 3231.]],

        [[3006., 6167., 9485., 6485., 3324.],
        [6477., 13276., 20401., 13936., 7137.],
        [7035., 14404., 22111., 15088., 7719.],
        [3777., 7727., 11852., 8081., 4131.]],

        [[3816., 7805., 11969., 8159., 4170.],
        [8151., 16660., 25531., 17392., 8883.],
        [8709., 17788., 27241., 18544., 9465.],
        [4641., 9473., 14498., 9863., 5031.]]]])
    */

    EXPECT_EQ(1632, in_err.GetReal(0, { 0, 0, 0 }));
    EXPECT_EQ(7444, in_err.GetReal(0, { 1, 1, 0 }));
    EXPECT_EQ(3816,  in_err.GetReal(1, { 0, 0, 2 }));
    EXPECT_EQ(5031, in_err.GetReal(1, { 4, 3, 2 }));

}

#endif


#if 0

TEST(NeuralNetConvolution2dTest, testNeuralNetConvolution2dCmp)
{
#if 0
    const int c_size = 31; // 2;
    const int x_size = 3;  // 3;
    const int y_size = 3;  //  3;
    const int n_size = 32; // 4;
    const int w_size = 28; // 5;
    const int h_size = 28; //6;
    const int batch_size = 256;//  128;
#else
    const int c_size = 2;
    const int x_size = 3;
    const int y_size = 3;
    const int n_size = 4;
    const int w_size = 5;
    const int h_size = 6;
    const int batch_size = 128;
#endif
    const int out_w_size = w_size - x_size + 1;
    const int out_h_size = h_size - y_size + 1;


    bb::NeuralNetDenseAffine<>          sub_affine(c_size * y_size * x_size, n_size);
    bb::NeuralNetGroup<>                sub_net;
    sub_net.AddLayer(&sub_affine);
    bb::NeuralNetConvolution2d<>  cnv0(&sub_net, c_size, w_size, h_size, n_size, x_size, y_size);
    bb::NeuralNetDenseConvolution<>     cnv1(          c_size, w_size, h_size, n_size, x_size, y_size);

    cnv0.SetBatchSize(batch_size);
    cnv1.SetBatchSize(batch_size);

    testSetupLayerBuffer(cnv0);
    testSetupLayerBuffer(cnv1);

    bb::NeuralNetOptimizerAdam<> optimizerAdam;
    cnv0.SetOptimizer(&optimizerAdam);
    cnv1.SetOptimizer(&optimizerAdam);


    std::mt19937_64 mt(1);
    std::uniform_int_distribution<int>  rand_dist(1, 16);

    for (int epoc = 0; epoc < 3; ++epoc) {
        // -------------------------------
        //  forward
        // -------------------------------

        // 係数を統一
        for (int n = 0; n < n_size; ++n) {
            for (int c = 0; c < c_size; ++c) {
                for (int y = 0; y < y_size; ++y) {
                    for (int x = 0; x < x_size; ++x) {
                        float W = (float)(rand_dist(mt));
                        sub_affine.W(n, (c*y_size + y)*x_size + x) = W;
                        cnv1.W(n, c, y, x) = W;
                    }
                }
            }
            float b = (float)(rand_dist(mt));
            sub_affine.b(n) = b;
            cnv1.b(n) = b;
        }

        // 入力を設定
        {
            auto in_sig_buf0 = cnv0.GetInputSignalBuffer();
            auto in_sig_buf1 = cnv1.GetInputSignalBuffer();
            for (int f = 0; f < batch_size; ++f) {
                for (int c = 0; c < c_size; ++c) {
                    for (int h = 0; h < h_size; ++h) {
                        for (int w = 0; w < w_size; ++w) {
                            float sig = (float)(rand_dist(mt));
                            in_sig_buf0.SetReal(f, (c*h_size + h)*w_size + w, sig);
                            in_sig_buf1.SetReal(f, (c*h_size + h)*w_size + w, sig);
                        }
                    }
                }
            }
        }

        cnv0.Forward();
        cnv1.Forward();

        // 出力を比較
        {
            auto out_sig_buf0 = cnv0.GetOutputSignalBuffer();
            auto out_sig_buf1 = cnv1.GetOutputSignalBuffer();
            for (int f = 0; f < batch_size; ++f) {
                for (int n = 0; n < c_size; ++n) {
                    for (int h = 0; h < out_h_size; ++h) {
                        for (int w = 0; w < out_w_size; ++w) {
                            auto sig0 = out_sig_buf0.GetReal(f, (n*out_h_size + h)*out_w_size + w);
                            auto sig1 = out_sig_buf1.GetReal(f, (n*out_h_size + h)*out_w_size + w);
                            EXPECT_EQ(sig0, sig1);
                        }
                    }
                }
            }
        }



        // -------------------------------
        //  backword
        // -------------------------------

        // 逆伝播誤差を設定
        auto out_err_buf0 = cnv0.GetOutputErrorBuffer();
        auto out_err_buf1 = cnv1.GetOutputErrorBuffer();
        for (int f = 0; f < batch_size; ++f) {
            for (int n = 0; n < c_size; ++n) {
                for (int h = 0; h < out_h_size; ++h) {
                    for (int w = 0; w < out_w_size; ++w) {
                        float err = (float)(rand_dist(mt));
                        out_err_buf0.SetReal(f, (n*out_h_size + h)*out_w_size + w, err);
                        out_err_buf1.SetReal(f, (n*out_h_size + h)*out_w_size + w, err);
                    }
                }
            }
        }

        cnv0.Backward();
        cnv1.Backward();

        // 伝播誤差を比較
        {
            auto in_err_buf0 = cnv0.GetInputErrorBuffer();
            auto in_err_buf1 = cnv1.GetInputErrorBuffer();
            for (int f = 0; f < batch_size; ++f) {
                for (int c = 0; c < c_size; ++c) {
                    for (int h = 0; h < h_size; ++h) {
                        for (int w = 0; w < w_size; ++w) {
                            float sig = (float)(rand_dist(mt));
                            auto err0 = in_err_buf0.GetReal(f, (c*h_size + h)*w_size + w);
                            auto err1 = in_err_buf1.GetReal(f, (c*h_size + h)*w_size + w);
                            EXPECT_EQ(err0, err1);
                        }
                    }
                }
            }
        }

        // 学習係数を比較
        for (int n = 0; n < n_size; ++n) {
            for (int c = 0; c < c_size; ++c) {
                for (int y = 0; y < y_size; ++y) {
                    for (int x = 0; x < x_size; ++x) {
                        auto dW0 = sub_affine.dW(n, (c*y_size + y)*x_size + x);
                        auto dW1 = cnv1.dW(n, c, y, x);
                        EXPECT_EQ(dW0, dW1);
                    }
                }
            }
            auto db0 = sub_affine.db(n);
            auto db1 = cnv1.db(n);
            EXPECT_EQ(db0, db1);
        }


        cnv0.Update();
        cnv1.Update();

        // 学習係数を比較
        for (int n = 0; n < n_size; ++n) {
            for (int c = 0; c < c_size; ++c) {
                for (int y = 0; y < y_size; ++y) {
                    for (int x = 0; x < x_size; ++x) {
                        auto W0 = sub_affine.W(n, (c*y_size + y)*x_size + x);
                        auto W1 = cnv1.W(n, c, y, x);
                        EXPECT_EQ(W0, W1);
                    }
                }
            }
            auto b0 = sub_affine.b(n);
            auto b1 = cnv1.b(n);
            EXPECT_EQ(b0, b1);
        }
    }
}


#endif

