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

#include <cereal/cereal.hpp>
#include <cereal/types/array.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/archives/json.hpp>

#include "bb/NeuralNet.h"
#include "bb/NeuralNetUtility.h"

#include "bb/NeuralNetSigmoid.h"
#include "bb/NeuralNetReLU.h"
#include "bb/NeuralNetSoftmax.h"
#include "bb/NeuralNetBinarize.h"
#include "bb/NeuralNetDropout.h"

#include "bb/NeuralNetSparseMicroMlp.h"
#include "bb/NeuralNetSparseMicroMlpDiscrete.h"

#include "bb/NeuralNetBinaryMultiplex.h"

#include "bb/NeuralNetBatchNormalization.h"

#include "bb/NeuralNetDenseAffine.h"
#include "bb/NeuralNetSparseAffine.h"
#include "bb/NeuralNetSparseBinaryAffine.h"

#include "bb/NeuralNetRealToBinary.h"
#include "bb/NeuralNetBinaryToReal.h"
#include "bb/NeuralNetBinaryLut6.h"

#include "bb/NeuralNetBinaryLut6VerilogXilinx.h"
#include "bb/NeuralNetBinaryLutVerilog.h"
#include "bb/NeuralNetOutputVerilog.h"

#include "bb/NeuralNetSparseAffineSigmoid.h"

#include "bb/NeuralNetOptimizerSgd.h"
#include "bb/NeuralNetOptimizerAdam.h"

#include "bb/NeuralNetDenseConvolution.h"
#include "bb/NeuralNetMaxPooling.h"

#include "bb/NeuralNetLossCrossEntropyWithSoftmax.h"
#include "bb/NeuralNetAccuracyCategoricalClassification.h"

#include "bb/NeuralNetLoweringConvolution.h"

#include "bb/ShuffleSet.h"

#include "bb/LoadMnist.h"
#include "bb/DataAugmentationMnist.h"
#include "bb/NeuralNetDenseToSparseAffine.h"



// LUT6入力のバイナリ版の力技学習  with BruteForce training
void MnistMlpLutDirect(int epoc_size, size_t mini_batch_size, std::string name)
{
    // run name
    std::string run_name = name; //  "MnistMlpLut";
    int         num_class = 10;
    int         max_train = -1;
    int         max_test = -1;

#ifdef _DEBUG
    std::cout << "!!!Debug mode!!!" << std::endl;
    max_train = 100;
    max_test = 50;
    mini_batch_size = 16;
#endif

    // load MNIST data
    auto train_data = bb::LoadMnist<>::Load(num_class, max_train, max_test);
    auto& x_train = train_data.x_train;
    auto& y_train = train_data.y_train;
    auto& x_test = train_data.x_test;
    auto& y_test = train_data.y_test;
    auto label_train = bb::OnehotToLabel<std::uint8_t>(y_train);
    auto label_test = bb::OnehotToLabel<std::uint8_t>(y_test);
    auto train_size = x_train.size();
    auto test_size = x_test.size();
    auto x_node_size = x_test[0].size();

    std::cout << "start : " << run_name << std::endl;

    std::mt19937_64 mt(1);

    // 学習時と評価時で多重化数(乱数を変えて複数枚通して集計できるようにする)を変える
    int train_mux_size = 3;
    int test_mux_size = 3;

    // define layer size
    size_t input_node_size = 28 * 28;
    size_t output_node_size = 10;
    size_t input_hmux_size = 1;
    size_t output_hmux_size = 3;

    size_t layer0_node_size = 360;
    size_t layer1_node_size = 60 * output_hmux_size;
    size_t layer2_node_size = output_node_size * output_hmux_size;

    // バイナリネットのGroup作成
    bb::NeuralNetBinaryLut6<>   bin_layer0_lut(input_node_size*input_hmux_size, layer0_node_size);
    bb::NeuralNetBinaryLut6<>   bin_layer1_lut(layer0_node_size, layer1_node_size);
    bb::NeuralNetBinaryLut6<>   bin_layer2_lut(layer1_node_size, layer2_node_size);
    bb::NeuralNetGroup<>        bin_group;
    bin_group.AddLayer(&bin_layer0_lut);
    bin_group.AddLayer(&bin_layer1_lut);
    bin_group.AddLayer(&bin_layer2_lut);

    // 多重化してパッキング
    bb::NeuralNetBinaryMultiplex<>  bin_mux(&bin_group, input_node_size, output_node_size, input_hmux_size, output_hmux_size);

    // ネット構築
    bb::NeuralNet<> net;
    net.AddLayer(&bin_mux);

    // 評価関数
    bb::NeuralNetAccuracyCategoricalClassification<>    accFunc(num_class);

    // 初期評価
    bin_mux.SetMuxSize(test_mux_size);  // 評価用の多重化数にスイッチ
    auto test_accuracy = net.RunCalculation(train_data.x_test, train_data.y_test, mini_batch_size, 0, &accFunc);
    std::cout << "initial test_accuracy : " << test_accuracy << std::endl;

    // 開始時間記録
    auto start_time = std::chrono::system_clock::now();

    // 学習ループ
    for (int epoc = 0; epoc < epoc_size; ++epoc) {
        int iteration = 0;
        for (size_t train_index = 0; train_index < train_size; train_index += mini_batch_size) {
            // 末尾のバッチサイズクリップ
            size_t batch_size = std::min(mini_batch_size, train_size - train_index);
            if (batch_size < mini_batch_size) { break; }

            // 小サイズで演算すると劣化するので末尾スキップ
            if (batch_size < mini_batch_size) {
                break;
            }

            // バッチサイズ設定
            bin_mux.SetMuxSize(train_mux_size); // 学習の多重化数にスイッチ
            net.SetBatchSize(batch_size);

            // データ格納
            auto in_sig_buf = net.GetInputSignalBuffer();
            for (size_t frame = 0; frame < batch_size; ++frame) {
                for (size_t node = 0; node < x_node_size; ++node) {
                    in_sig_buf.Set<float>(frame, node, x_train[train_index + frame][node]);
                }
            }

            // 予測
            net.Forward(true);

            // バイナリ版フィードバック(力技学習)
            while (bin_mux.Feedback(bin_mux.GetOutputOnehotLoss<std::uint8_t, 10>(label_train, train_index)))
                ;

            //      while (bin_mux.Feedback(bin_mux.GetOutputOnehotLoss(y_train, train_index)))
            //          ;

            //      while (bin_mux.Feedback(bin_mux.CalcLoss(y_train, train_index)))
            //          ;

            // 途中評価
            bin_mux.SetMuxSize(test_mux_size);  // 評価用の多重化数にスイッチ
            auto test_accuracy = net.RunCalculation(x_test, y_test, mini_batch_size, 0, &accFunc);

            // 進捗表示
            auto progress = train_index + batch_size;
            auto rate = progress * 100 / train_size;
            std::cout << "[" << rate << "% (" << progress << "/" << train_size << ")]";
            std::cout << "  test_accuracy : " << test_accuracy << "                  ";
            std::cout << "\r" << std::flush;
        }

        // 評価
        bin_mux.SetMuxSize(test_mux_size);  // 評価用の多重化数にスイッチ
        auto test_accuracy = net.RunCalculation(x_test, y_test, mini_batch_size, 0, &accFunc);
        auto train_accuracy = net.RunCalculation(x_train, y_train, mini_batch_size, 0, &accFunc);
        auto now_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time).count() / 1000.0;
        std::cout << now_time << "s " << "epoc[" << epoc + 1 << "]"
            << "  test_accuracy : " << test_accuracy
            << "  train_accuracy : " << train_accuracy << std::endl;

        // Shuffle
        bb::ShuffleDataSet(mt(), x_train, y_train, label_train);
    }

    {
        // Write RTL
        std::ofstream ofs("lut_net.v");
        bb::NeuralNetBinaryLut6VerilogXilinx(ofs, bin_layer0_lut, "lutnet_layer0");
        bb::NeuralNetBinaryLut6VerilogXilinx(ofs, bin_layer1_lut, "lutnet_layer1");
        bb::NeuralNetBinaryLut6VerilogXilinx(ofs, bin_layer2_lut, "lutnet_layer2");
    }

    std::cout << "end\n" << std::endl;
}

// MNIST Multilayer perceptron with LUT networks
void MnistSparseAffine(int epoc_size, size_t max_batch_size, bool binary_mode, std::string name)
{
    // parameter
    std::string run_name = name; //  "MnistMlpLut2";
    int         num_class = 10;

    // load MNIST data
    auto td = bb::LoadMnist<>::Load();

    // 学習時と評価時で多重化数(乱数を変えて複数枚通して集計できるようにする)を変える
    int train_mux_size = 1;
    int test_mux_size = 3;

    // define layer size
    size_t input_node_size = 28 * 28;
    size_t output_node_size = 10;
    size_t input_hmux_size = 1;
    size_t output_hmux_size = 3;

    size_t layer0_node_size = 360;
    size_t layer1_node_size = 60 * output_hmux_size;
    size_t layer2_node_size = output_node_size * output_hmux_size;

    // build layer
    bb::NeuralNetSparseAffineSigmoid<6> layer0_lut(input_node_size*input_hmux_size, layer0_node_size);
    bb::NeuralNetSparseAffineSigmoid<6> layer1_lut(layer0_node_size, layer1_node_size);
    bb::NeuralNetSparseAffineSigmoid<6> layer2_lut(layer1_node_size, layer2_node_size);
    bb::NeuralNetGroup<>        bin_group;
    bin_group.AddLayer(&layer0_lut);
    bin_group.AddLayer(&layer1_lut);
    bin_group.AddLayer(&layer2_lut);

    // 多重化してパッキング
    bb::NeuralNetBinaryMultiplex<float> bin_mux(&bin_group, input_node_size, output_node_size, input_hmux_size, output_hmux_size);

    // ネット構築
    bb::NeuralNet<> net;
    net.AddLayer(&bin_mux);

    bin_mux.SetMuxSize(test_mux_size);

    // set optimizer
    bb::NeuralNetOptimizerAdam<> optimizer;
    net.SetOptimizer(&optimizer);

    // set binary mode
    net.SetBinaryMode(binary_mode);
    std::cout << "binary mode : " << binary_mode << std::endl;

    // run fitting
    bb::NeuralNetLossCrossEntropyWithSoftmax<>          loss_func;
    bb::NeuralNetAccuracyCategoricalClassification<>    acc_func(num_class);
    net.Fitting(run_name, td, epoc_size, max_batch_size, &acc_func, &loss_func, true, true, true);
}




// MNIST Multilayer perceptron with LUT networks
template <int N=16>
void MnistMlpLutN(int epoc_size, size_t max_batch_size, bool binary_mode, std::string name)
{
    // parameter
    std::string run_name = name; //  "MnistMlpLut2";
    int         num_class = 10;

    // load MNIST data
    auto td = bb::LoadMnist<>::Load();

    // 学習時と評価時で多重化数(乱数を変えて複数枚通して集計できるようにする)を変える
    int train_mux_size = 1;
    int test_mux_size = 3;

    // define layer size
    size_t input_node_size = 28 * 28;
    size_t output_node_size = 10;
    size_t input_hmux_size = 1;
    size_t output_hmux_size = 3;

    size_t layer0_node_size = 360;
    size_t layer1_node_size = 60 * output_hmux_size;
    size_t layer2_node_size = output_node_size * output_hmux_size;

    // build layer
    bb::NeuralNetSparseMicroMlpDiscrete<6, N>   layer0_lut(input_node_size*input_hmux_size, layer0_node_size);
    bb::NeuralNetSparseMicroMlpDiscrete<6, N>   layer1_lut(layer0_node_size, layer1_node_size);
    bb::NeuralNetSparseMicroMlpDiscrete<6, N>   layer2_lut(layer1_node_size, layer2_node_size);
    bb::NeuralNetGroup<>        bin_group;
    bin_group.AddLayer(&layer0_lut);
    bin_group.AddLayer(&layer1_lut);
    bin_group.AddLayer(&layer2_lut);

    // 多重化してパッキング
    bb::NeuralNetBinaryMultiplex<float> bin_mux(&bin_group, input_node_size, output_node_size, input_hmux_size, output_hmux_size);

    // ネット構築
    bb::NeuralNet<> net;
    net.AddLayer(&bin_mux);

    bin_mux.SetMuxSize(test_mux_size);

    // set optimizer
    bb::NeuralNetOptimizerAdam<> optimizer;
    net.SetOptimizer(&optimizer);

    // set binary mode
    net.SetBinaryMode(binary_mode);
    std::cout << "binary mode : " << binary_mode << std::endl;

    // run fitting
    bb::NeuralNetLossCrossEntropyWithSoftmax<>          loss_func;
    bb::NeuralNetAccuracyCategoricalClassification<>    acc_func(num_class);
    net.Fitting(run_name, td, epoc_size, max_batch_size, &acc_func, &loss_func, true, true, true);
}



// メイン関数
int evaluate_micro_mlp(void)
{
    omp_set_num_threads(4);

    MnistSparseAffine(16, 128, true, "MnistSparseLut");
    getchar();

    MnistMlpLutN<1> (16, 128, true, "MnistMlpLutN_1");

    MnistMlpLutDirect(16, 8192,     "MnistMlpLutDirect");
    MnistMlpLutN<64>(16, 128, true, "MnistMlpLutN_64");
    MnistMlpLutN<32>(16, 128, true, "MnistMlpLutN_32");
    MnistMlpLutN<16>(16, 128, true, "MnistMlpLutN_16");
    MnistMlpLutN<8> (16, 128, true, "MnistMlpLutN_8");
    MnistMlpLutN<4> (16, 128, true, "MnistMlpLutN_4");
    
    getchar();
    
    return 0;
}


