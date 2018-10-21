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

#include "bb/NeuralNetBinaryMultiplex.h"

#include "bb/NeuralNetBatchNormalization.h"

#include "bb/NeuralNetAffine.h"
#include "bb/NeuralNetSparseAffine.h"
#include "bb/NeuralNetSparseBinaryAffine.h"

#include "bb/NeuralNetRealToBinary.h"
#include "bb/NeuralNetBinaryToReal.h"
#include "bb/NeuralNetBinaryLut6.h"
#include "bb/NeuralNetBinaryLut6VerilogXilinx.h"

#include "bb/NeuralNetSparseAffineSigmoid.h"

#include "bb/NeuralNetOptimizerSgd.h"
#include "bb/NeuralNetOptimizerAdam.h"

#include "bb/NeuralNetConvolution.h"
#include "bb/NeuralNetMaxPooling.h"
#include "bb/NeuralNetConvolutionPack.h"

#include "bb/NeuralNetLossCrossEntropyWithSoftmax.h"
#include "bb/NeuralNetAccuracyCategoricalClassification.h"

#include "bb/ShuffleSet.h"

#include "bb/LoadMnist.h"



void MnistDenseAffineReal(int epoc_size, size_t max_batch_size);
void MnistDenseAffineBinary(int epoc_size, size_t max_batch_size, bool binary_mode = true);

void MnistSparseAffineReal(int epoc_size, size_t max_batch_size);
void MnistSparseAffineBinary(int epoc_size, size_t max_batch_size, bool binary_mode = true);
void MnistSparseAffineLut6(int epoc_size, size_t max_batch_size);
void MnistSparseAffineBinToLut(int bin_epoc_size, size_t bin_max_batch_size, int lut_epoc_size, size_t lut_max_batch_size);

void MnistDenseSimpleConvolutionReal(int epoc_size, size_t max_batch_size);
void MnistDenseSimpleConvolutionBinary(int epoc_size, size_t max_batch_size, bool binary_mode = true);

void MnistSparseSimpleConvolutionBinary(int epoc_size, size_t max_batch_size, bool binary_mode = true);
void MnistSparseSimpleConvolutionBinToLut(int bin_epoc_size, size_t bin_max_batch_size, int lut_epoc_size, size_t lut_max_batch_size);

void MnistSparseSimpleConvolutionLut(int bin_epoc_size, size_t bin_max_batch_size);


// メイン関数
int main()
{
	omp_set_num_threads(6);

	// Dense Affine
#if 0
	MnistDenseAffineReal(16, 128);
#endif

#if 0
	MnistDenseAffineBinary(16, 128);
#endif


	// Sparse Affine
#if 0
	MnistSparseAffineReal(16, 128);
#endif

#if 0
	MnistSparseAffineBinary(16, 128);
#endif

#if 0
	MnistSparseAffineLut6(8, 8192);
#endif

#if 1
	MnistSparseAffineBinToLut(64, 128, 8, 8192);
#endif


	// Simple Convolution
#if 0
	MnistDenseSimpleConvolutionReal(16, 128);
#endif

#if 0
	MnistDenseSimpleConvolutionBinary(16, 128);
#endif

#if 0
	MnistSparseSimpleConvolutionBinary(16, 128);
#endif

#if 0
	MnistSparseSimpleConvolutionBinToLut(16, 128, 2, 256);
#endif

#if 0
	MnistSparseSimpleConvolutionLut(16, 256);
#endif

	return 0;
}



// DenseAffine Real network
void MnistDenseAffineReal(int epoc_size, size_t max_batch_size)
{
	// run name
	std::string run_name = "MnistDenseAffineReal";
	int			num_class = 10;

	// load MNIST data
	auto train_data = bb::LoadMnist<>::Load();
	
	// build layer
	bb::NeuralNetAffine<>  layer0_affine(28 * 28, 256);
	bb::NeuralNetSigmoid<> layer0_sigmoid(256);
	bb::NeuralNetAffine<>  layer1_affine(256, 10);

	// build network
	bb::NeuralNet<> net;
	net.AddLayer(&layer0_affine);
	net.AddLayer(&layer0_sigmoid);
	net.AddLayer(&layer1_affine);
	// loss function has softmax layer

	// set optimizer
	bb::NeuralNetOptimizerAdam<> optimizer;
	net.SetOptimizer(&optimizer);

	// run fitting
	bb::NeuralNetLossCrossEntropyWithSoftmax<>			lossFunc;
	bb::NeuralNetAccuracyCategoricalClassification<>	accFunc(num_class);		
	net.Fitting(run_name, train_data, epoc_size, max_batch_size, &accFunc, &lossFunc, true, false);
}

// Binary DenseAffine network
void MnistDenseAffineBinary(int epoc_size, size_t max_batch_size, bool binary_mode)
{
	// run name
	std::string run_name = "MnistDenseAffineBinary";

	// parameter
	int			num_class       = 10;
	int			binary_mux_size = 3;

	// load MNIST data
	auto train_data = bb::LoadMnist<>::Load();

	// build layer
	bb::NeuralNetAffine<>				layer0_affine(28 * 28, 256);
	bb::NeuralNetBatchNormalization<>	layer0_batch_norm(256);
	bb::NeuralNetSigmoid<>				layer0_activation(256);
	bb::NeuralNetAffine<>				layer1_affine(256, 10);
	bb::NeuralNetBatchNormalization<>	layer1_batch_norm(10);
	bb::NeuralNetSigmoid<>				layer1_activation(10);

	bb::NeuralNetGroup<>				mux_group;
	mux_group.AddLayer(&layer0_affine);
	mux_group.AddLayer(&layer0_batch_norm);
	mux_group.AddLayer(&layer0_activation);
	mux_group.AddLayer(&layer1_affine);
	mux_group.AddLayer(&layer1_batch_norm);
	mux_group.AddLayer(&layer1_activation);

	bb::NeuralNetBinaryMultiplex<float>	bin_mux(&mux_group, 28*28, 10);

	// build network
	bb::NeuralNet<> net;
	net.AddLayer(&bin_mux);

	// set optimizer
	bb::NeuralNetOptimizerAdam<> optimizer;
	net.SetOptimizer(&optimizer);

	// set binary mode
	net.SetBinaryMode(binary_mode);
	std::cout << "binary mode : " << binary_mode << std::endl;

	// set multiplexing size
	bin_mux.SetMuxSize(binary_mux_size);

	// run fitting
	bb::NeuralNetLossCrossEntropyWithSoftmax<>			lossFunc;
	bb::NeuralNetAccuracyCategoricalClassification<>	accFunc(num_class);
	net.Fitting(run_name, train_data, epoc_size, max_batch_size, &accFunc, &lossFunc, true, false);
}



// Sparse Affine network
void MnistSparseAffineReal(int epoc_size, size_t max_batch_size)
{
	// run name
	std::string run_name = "MnistSparseAffineReal";
	int			num_class = 10;

	// load MNIST data
	auto train_data = bb::LoadMnist<>::Load();

	// build layer
	bb::NeuralNetSparseAffine<6> layer0_affine(28 * 28, 360);
	bb::NeuralNetSigmoid<>		 layer0_sigmoid(360);
	bb::NeuralNetSparseAffine<6> layer1_affine(360, 60);
	bb::NeuralNetSigmoid<>		 layer1_sigmoid(60);
	bb::NeuralNetSparseAffine<6> layer2_affine(60, 10);
	// loss function has softmax layer

	// build network
	bb::NeuralNet<> net;
	net.AddLayer(&layer0_affine);
	net.AddLayer(&layer0_sigmoid);
	net.AddLayer(&layer1_affine);
	net.AddLayer(&layer1_sigmoid);
	net.AddLayer(&layer2_affine);

	// set optimizer
	bb::NeuralNetOptimizerAdam<> optimizer;
	net.SetOptimizer(&optimizer);

	// run fitting
	bb::NeuralNetLossCrossEntropyWithSoftmax<>			lossFunc;
	bb::NeuralNetAccuracyCategoricalClassification<>	accFunc(num_class);
	net.Fitting(run_name, train_data, epoc_size, max_batch_size, &accFunc, &lossFunc);
}

// Binary Sparse-Affine network
void MnistSparseAffineBinary(int epoc_size, size_t max_batch_size, bool binary_mode)
{
	// parameter
	std::string run_name = "MnistSparseAffineBinary";
	int			num_class = 10;
	int			binary_mux_size = 1;

	// load MNIST data
	auto train_data = bb::LoadMnist<>::Load();

	// build layer
	bb::NeuralNetSparseAffine<6>		layer0_affine(28*28, 360);
	bb::NeuralNetBatchNormalization<>	layer0_batch_norm(360);
	bb::NeuralNetSigmoid<>				layer0_activation(360);

	bb::NeuralNetSparseAffine<6>		layer1_affine(360, 60);
	bb::NeuralNetBatchNormalization<>	layer1_batch_norm(60);
	bb::NeuralNetSigmoid<>				layer1_activation(60);

	bb::NeuralNetSparseAffine<6>		layer2_affine(60, 10);
	bb::NeuralNetBatchNormalization<>	layer2_batch_norm(10);
	bb::NeuralNetSigmoid<>				layer2_activation(10);

	bb::NeuralNetGroup<>				mux_group;
	mux_group.AddLayer(&layer0_affine);
	mux_group.AddLayer(&layer0_batch_norm);
	mux_group.AddLayer(&layer0_activation);
	mux_group.AddLayer(&layer1_affine);
	mux_group.AddLayer(&layer1_batch_norm);
	mux_group.AddLayer(&layer1_activation);
	mux_group.AddLayer(&layer2_affine);
	mux_group.AddLayer(&layer2_batch_norm);
	mux_group.AddLayer(&layer2_activation);

	bb::NeuralNetBinaryMultiplex<float>	bin_mux(&mux_group, 28 * 28, 10);

	// build network
	bb::NeuralNet<> net;
	net.AddLayer(&bin_mux);

	// set optimizer
	bb::NeuralNetOptimizerAdam<> optimizer;
	net.SetOptimizer(&optimizer);

	// set binary mode
	net.SetBinaryMode(binary_mode);
	std::cout << "binary mode : " << binary_mode << std::endl;

	// set multiplexing size
	bin_mux.SetMuxSize(binary_mux_size);

	// run fitting
	bb::NeuralNetLossCrossEntropyWithSoftmax<>			lossFunc;
	bb::NeuralNetAccuracyCategoricalClassification<>	accFunc(num_class);
	net.Fitting(run_name, train_data, epoc_size, max_batch_size, &accFunc, &lossFunc, true, false);
}

// LUT6入力のバイナリ版の力技学習  with BruteForce training
void MnistSparseAffineLut6(int epoc_size, size_t max_batch_size)
{
	// run name
	std::string run_name = "MnistSparseAffineLut6";
	int			num_class = 10;
	int			max_train = -1;
	int			max_test = -1;

#ifdef _DEBUG
	std::cout << "!!!Debug mode!!!" << std::endl;
	max_train = 100;
	max_test = 50;
	max_batch_size = 16;
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

	// バイナリネットのGroup作成
	bb::NeuralNetBinaryLut6<>	bin_layer0_lut(input_node_size*input_hmux_size, layer0_node_size);
	bb::NeuralNetBinaryLut6<>	bin_layer1_lut(layer0_node_size, layer1_node_size);
	bb::NeuralNetBinaryLut6<>	bin_layer2_lut(layer1_node_size, layer2_node_size);
	bb::NeuralNetGroup<>		bin_group;
	bin_group.AddLayer(&bin_layer0_lut);
	bin_group.AddLayer(&bin_layer1_lut);
	bin_group.AddLayer(&bin_layer2_lut);

	// 多重化してパッキング
	bb::NeuralNetBinaryMultiplex<>	bin_mux(&bin_group, input_node_size, output_node_size, input_hmux_size, output_hmux_size);

	// ネット構築
	bb::NeuralNet<> net;
	net.AddLayer(&bin_mux);

	// 評価関数
	bb::NeuralNetAccuracyCategoricalClassification<>	accFunc(num_class);

	// 初期評価
	bin_mux.SetMuxSize(test_mux_size);	// 評価用の多重化数にスイッチ
	auto test_accuracy = net.RunCalculation(train_data.x_test, train_data.y_test, max_batch_size, 0, &accFunc);
	std::cout << "initial test_accuracy : " << test_accuracy << std::endl;

	// 開始時間記録
	auto start_time = std::chrono::system_clock::now();

	// 学習ループ
	for (int epoc = 0; epoc < epoc_size; ++epoc) {
		int iteration = 0;
		for (size_t train_index = 0; train_index < train_size; train_index += max_batch_size) {
			// 末尾のバッチサイズクリップ
			size_t batch_size = std::min(max_batch_size, train_size - train_index);
			if (batch_size < max_batch_size) { break; }

			// 小サイズで演算すると劣化するので末尾スキップ
			if (batch_size < max_batch_size) {
				break;
			}

			// バッチサイズ設定
			bin_mux.SetMuxSize(train_mux_size);	// 学習の多重化数にスイッチ
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
	
	//		while (bin_mux.Feedback(bin_mux.GetOutputOnehotLoss(y_train, train_index)))
	//			;
	
	//		while (bin_mux.Feedback(bin_mux.CalcLoss(y_train, train_index)))
	//			;

			// 途中評価
			bin_mux.SetMuxSize(test_mux_size);	// 評価用の多重化数にスイッチ
			auto test_accuracy = net.RunCalculation(x_test, y_test, max_batch_size, 0, &accFunc);

			// 進捗表示
			auto progress = train_index + batch_size;
			auto rate = progress * 100 / train_size;
			std::cout << "[" << rate << "% (" << progress << "/" << train_size << ")]";
			std::cout << "  test_accuracy : " << test_accuracy << "                  ";
			std::cout << "\r" << std::flush;
		}

		// 評価
		bin_mux.SetMuxSize(test_mux_size);	// 評価用の多重化数にスイッチ
		auto test_accuracy = net.RunCalculation(x_test, y_test, max_batch_size, 0, &accFunc);
		auto train_accuracy = net.RunCalculation(x_train,y_train, max_batch_size, 0, &accFunc);
		auto now_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time).count() / 1000.0;
		std::cout << now_time << "s " << "epoc[" << epoc+1 << "]"
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

// Binary-Network copy to LUT-Network
void MnistSparseAffineBinToLut(int bin_epoc_size, size_t bin_max_batch_size, int lut_epoc_size, size_t lut_max_batch_size)
{
	// parameter
	std::string run_name = "MnistSparseAffineBinToLut";
	int			num_class = 10;

	// load MNIST data
	auto train_data = bb::LoadMnist<>::Load();

	// define layer size
	size_t		input_node_size = 28 * 28;
	size_t		output_node_size = 10;
	size_t		input_hmux_size = 1;
	size_t		output_hmux_size = 3;

	size_t		layer0_node_size = 360;
	size_t		layer1_node_size = 60 * output_hmux_size;
	size_t		layer2_node_size = output_node_size * output_hmux_size;



	// -------- Binary-Network --------

	int	bin_mux_size = 1;

	// build layer
	bb::NeuralNetSparseBinaryAffine<>	bin_layer0_affine(input_node_size*input_hmux_size, layer0_node_size);
	bb::NeuralNetSparseBinaryAffine<>	bin_layer1_affine(layer0_node_size, layer1_node_size);
	bb::NeuralNetSparseBinaryAffine<>	bin_layer2_affine(layer1_node_size, layer2_node_size);

	bb::NeuralNetGroup<>				bin_mux_group;
	bin_mux_group.AddLayer(&bin_layer0_affine);
	bin_mux_group.AddLayer(&bin_layer1_affine);
	bin_mux_group.AddLayer(&bin_layer2_affine);

	bb::NeuralNetBinaryMultiplex<float>	bin_mux(&bin_mux_group, input_node_size, output_node_size, input_hmux_size, output_hmux_size);

	// build network
	bb::NeuralNet<> bin_net;
	bin_net.AddLayer(&bin_mux);

	// set optimizer
	bb::NeuralNetOptimizerAdam<> optimizer;
	bin_net.SetOptimizer(&optimizer);

	// set binary mode
	bin_net.SetBinaryMode(true);

	// set multiplexing size
	bin_mux.SetMuxSize(bin_mux_size);

	// run fitting
	bb::NeuralNetLossCrossEntropyWithSoftmax<>			bin_loss_func;
	bb::NeuralNetAccuracyCategoricalClassification<>	bin_acc_func(num_class);
	bin_net.Fitting(run_name, train_data, bin_epoc_size, bin_max_batch_size, &bin_acc_func, &bin_loss_func, true, false);



	// -------- LUT-Network --------

	// load MNIST data
	//	train_data = bb::LoadMnist<>::Load(num_class);
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
	int lut_train_mux_size = 1;
	int lut_test_mux_size = 3;

	// バイナリネットのGroup作成
	bb::NeuralNetBinaryLut6<>	lut_layer0_lut(input_node_size*input_hmux_size, layer0_node_size);
	bb::NeuralNetBinaryLut6<>	lut_layer1_lut(layer0_node_size, layer1_node_size);
	bb::NeuralNetBinaryLut6<>	lut_layer2_lut(layer1_node_size, layer2_node_size);
	bb::NeuralNetGroup<>		lut_mux_group;
	lut_mux_group.AddLayer(&lut_layer0_lut);
	lut_mux_group.AddLayer(&lut_layer1_lut);
	lut_mux_group.AddLayer(&lut_layer2_lut);

	// 多重化してパッキング
	bb::NeuralNetBinaryMultiplex<>	lut_mux(&lut_mux_group, input_node_size, output_node_size, input_hmux_size, output_hmux_size);

	// ネット構築
	bb::NeuralNet<> lut_net;
	lut_net.AddLayer(&lut_mux);

	// 評価関数
	bb::NeuralNetAccuracyCategoricalClassification<>	lut_acc_func(num_class);


	// copy
	std::cout << "[parameter copy] Binary-Neteork -> LUT-Network" << std::endl;
	lut_layer0_lut.ImportLayer(bin_layer0_affine);
	lut_layer1_lut.ImportLayer(bin_layer1_affine);
	lut_layer2_lut.ImportLayer(bin_layer2_affine);


	// 初期評価
	lut_mux.SetMuxSize(lut_test_mux_size);	// 評価用の多重化数にスイッチ
	auto test_accuracy = lut_net.RunCalculation(train_data.x_test, train_data.y_test, lut_max_batch_size, 0, &lut_acc_func);
	std::cout << "initial test_accuracy : " << test_accuracy << std::endl;

	// 開始時間記録
	auto start_time = std::chrono::system_clock::now();

	// 学習ループ
	for (int epoc = 0; epoc < lut_epoc_size; ++epoc) {
		int iteration = 0;
		for (size_t train_index = 0; train_index < train_size; train_index += lut_max_batch_size) {
			// 末尾のバッチサイズクリップ
			size_t batch_size = std::min(lut_max_batch_size, train_size - train_index);
			if (batch_size < lut_max_batch_size) { break; }

			// 小サイズで演算すると劣化するので末尾スキップ
			if (batch_size < lut_max_batch_size) {
				break;
			}

			// バッチサイズ設定
			lut_mux.SetMuxSize(lut_train_mux_size);	// 学習の多重化数にスイッチ
			lut_net.SetBatchSize(batch_size);

			// データ格納
			auto in_sig_buf = lut_net.GetInputSignalBuffer();
			for (size_t frame = 0; frame < batch_size; ++frame) {
				for (size_t node = 0; node < x_node_size; ++node) {
					in_sig_buf.Set<float>(frame, node, x_train[train_index + frame][node]);
				}
			}

			// 予測
			lut_net.Forward(true);

			// バイナリ版フィードバック(力技学習)
			while (lut_mux.Feedback(lut_mux.GetOutputOnehotLoss<std::uint8_t, 10>(label_train, train_index)))
				;

			//		while (bin_mux.Feedback(bin_mux.GetOutputOnehotLoss(y_train, train_index)))
			//			;

			//		while (bin_mux.Feedback(bin_mux.CalcLoss(y_train, train_index)))
			//			;

			// 途中評価
			lut_mux.SetMuxSize(lut_test_mux_size);	// 評価用の多重化数にスイッチ
			auto test_accuracy = lut_net.RunCalculation(x_test, y_test, lut_max_batch_size, 0, &lut_acc_func);

			// 進捗表示
			auto progress = train_index + batch_size;
			auto rate = progress * 100 / train_size;
			std::cout << "[" << rate << "% (" << progress << "/" << train_size << ")]";
			std::cout << "  test_accuracy : " << test_accuracy << "                  ";
			std::cout << "\r" << std::flush;
		}

		// 評価
		lut_mux.SetMuxSize(lut_test_mux_size);	// 評価用の多重化数にスイッチ
		auto test_accuracy = lut_net.RunCalculation(x_test, y_test, lut_max_batch_size, 0, &lut_acc_func);
		auto train_accuracy = lut_net.RunCalculation(x_train, y_train, lut_max_batch_size, 0, &lut_acc_func);
		auto now_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time).count() / 1000.0;
		std::cout << now_time << "s " << "epoc[" << epoc << "]"
			<< "  test_accuracy : " << test_accuracy
			<< "  train_accuracy : " << train_accuracy << std::endl;

		// Shuffle
		bb::ShuffleDataSet(mt(), x_train, y_train, label_train);
	}

	{
		// Write RTL
		std::ofstream ofs("lut_net.v");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, lut_layer0_lut, "lutnet_layer0");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, lut_layer1_lut, "lutnet_layer1");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, lut_layer2_lut, "lutnet_layer2");
	}

	std::cout << "end\n" << std::endl;
}



// DenseSimpleConvolution network
void MnistDenseSimpleConvolutionReal(int epoc_size, size_t max_batch_size)
{
	// run name
	std::string run_name = "MnistDenseSimpleConvolutionReal";
	int			num_class = 10;

	// load MNIST data
	auto train_data = bb::LoadMnist<>::Load();

	// build layer
	bb::NeuralNetConvolution<>  layer0_conv(1, 28, 28, 16, 3, 3);	// c:1  w:28 h:28  --(filter:3x3)--> c:16 w:26 h:26 
	bb::NeuralNetConvolution<>  layer1_conv(16, 26, 26, 16, 3, 3);	// c:16 w:26 h:26  --(filter:3x3)--> c:16 w:24 h:24 
	bb::NeuralNetMaxPooling<>	layer2_maxpol(16, 24, 24, 2, 2);	// c:16 w:24 h:24  --(filter:2x2)--> c:16 w:12 h:12 
	bb::NeuralNetAffine<>		layer3_affine(16 * 12 * 12, 128);
	bb::NeuralNetSigmoid<>		layer4_sigmoid(128);
	bb::NeuralNetAffine<>		layer5_affine(128, 10);

	// build network
	bb::NeuralNet<> net;
	net.AddLayer(&layer0_conv);
	net.AddLayer(&layer1_conv);
	net.AddLayer(&layer2_maxpol);
	net.AddLayer(&layer3_affine);
	net.AddLayer(&layer4_sigmoid);
	net.AddLayer(&layer5_affine);

	// set optimizer
	bb::NeuralNetOptimizerAdam<> optimizerAdam;
	net.SetOptimizer(&optimizerAdam);

	// run fitting
	bb::NeuralNetLossCrossEntropyWithSoftmax<>			lossFunc;
	bb::NeuralNetAccuracyCategoricalClassification<>	accFunc(num_class);
	net.Fitting(run_name, train_data, epoc_size, max_batch_size, &accFunc, &lossFunc, true, false);
}


void MnistDenseSimpleConvolutionBinary(int epoc_size, size_t max_batch_size, bool binary_mode)
{
	// run name
	std::string run_name = "MnistDenseSimpleConvolutionBinary";
	int			num_class = 10;

	// load MNIST data
	auto train_data = bb::LoadMnist<>::Load();

	// build layer
	bb::NeuralNetConvolution<>			layer0_conv(1, 28, 28, 16, 3, 3);	// c:1  w:28 h:28  --(filter:3x3)--> c:16 w:26 h:26 
	bb::NeuralNetBatchNormalization<>	layer0_batch_norm(16 * 26 * 26);
	bb::NeuralNetSigmoid<>				layer0_activation(16 * 26 * 26);
	bb::NeuralNetConvolution<>			layer1_conv(16, 26, 26, 16, 3, 3);	// c:16 w:26 h:26  --(filter:3x3)--> c:16 w:24 h:24 
	bb::NeuralNetBatchNormalization<>	layer1_batch_norm(16 * 24 * 24);
	bb::NeuralNetSigmoid<>				layer1_activation(16 * 24 * 24);
	bb::NeuralNetMaxPooling<>			layer2_maxpol(16, 24, 24, 2, 2);	// c:16 w:24 h:24  --(filter:2x2)--> c:16 w:12 h:12 
	bb::NeuralNetAffine<>				layer3_affine(16 * 12 * 12, 180);
	bb::NeuralNetBatchNormalization<>	layer3_batch_norm(180);
	bb::NeuralNetSigmoid<>				layer3_activation(180);
	bb::NeuralNetAffine<>				layer4_affine(180, 30);
	bb::NeuralNetBatchNormalization<>	layer4_batch_norm(30);
	bb::NeuralNetSigmoid<>				layer4_activation(30);

	bb::NeuralNetGroup<>				mux_group;
	mux_group.AddLayer(&layer0_conv);
	mux_group.AddLayer(&layer0_batch_norm);
	mux_group.AddLayer(&layer0_activation);
	mux_group.AddLayer(&layer1_conv);
	mux_group.AddLayer(&layer1_batch_norm);
	mux_group.AddLayer(&layer1_activation);
	mux_group.AddLayer(&layer2_maxpol);
	mux_group.AddLayer(&layer3_affine);
	mux_group.AddLayer(&layer3_batch_norm);
	mux_group.AddLayer(&layer3_activation);
	mux_group.AddLayer(&layer4_affine);
	mux_group.AddLayer(&layer4_batch_norm);
	mux_group.AddLayer(&layer4_activation);

	bb::NeuralNetBinaryMultiplex<float>	bin_mux(&mux_group, 28 * 28, 10, 1, 3);
	
	// build network
	bb::NeuralNet<> net;
	net.AddLayer(&bin_mux);

	// set optimizer
	bb::NeuralNetOptimizerAdam<> optimizerAdam;
	net.SetOptimizer(&optimizerAdam);

	// set binary mode
	net.SetBinaryMode(binary_mode);
	std::cout << "binary mode : " << binary_mode << std::endl;

	// set multiplexing size
	bin_mux.SetMuxSize(1);

	// run fitting
	bb::NeuralNetLossCrossEntropyWithSoftmax<>			lossFunc;
	bb::NeuralNetAccuracyCategoricalClassification<>	accFunc(num_class);
	net.Fitting(run_name, train_data, epoc_size, max_batch_size, &accFunc, &lossFunc, true, false);
}


void MnistSparseSimpleConvolutionBinary(int epoc_size, size_t max_batch_size, bool binary_mode)
{
	// run name
	std::string run_name = "MnistSparseSimpleConvolutionBinary";
	int			num_class = 10;

	// load MNIST data
	auto train_data = bb::LoadMnist<>::Load();


	// Conv用subネット構築 (3x3)
	bb::NeuralNetSparseBinaryAffine<>	sub0_affine0(1 * 3 * 3, 96);
	bb::NeuralNetSparseBinaryAffine<>	sub0_affine1(96, 16);
	bb::NeuralNetGroup<>				sub0_net;
	sub0_net.AddLayer(&sub0_affine0);
	sub0_net.AddLayer(&sub0_affine1);

	// Conv用subネット構築 (3x3)
	bb::NeuralNetSparseBinaryAffine<>	sub1_affine0(16 * 3 * 3, 256);
	bb::NeuralNetSparseBinaryAffine<>	sub1_affine1(256, 96);
	bb::NeuralNetSparseBinaryAffine<>	sub1_affine2(96, 16);
	bb::NeuralNetGroup<>				sub1_net;
	sub1_net.AddLayer(&sub1_affine0);
	sub1_net.AddLayer(&sub1_affine1);
	sub1_net.AddLayer(&sub1_affine2);

	bb::NeuralNetConvolutionPack<>		layer0_conv(&sub0_net, 1, 28, 28, 16, 3, 3);
	bb::NeuralNetConvolutionPack<>		layer1_conv(&sub1_net, 16, 26, 26, 16, 3, 3);
	bb::NeuralNetMaxPooling<>			layer2_maxpol(16, 24, 24, 2, 2);
	bb::NeuralNetSparseBinaryAffine<>	layer3_affine(16 * 12 * 12, 180);
	bb::NeuralNetSparseBinaryAffine<>	layer4_affine(180, 30);

	bb::NeuralNetGroup<>				mux_group;
	mux_group.AddLayer(&layer0_conv);
	mux_group.AddLayer(&layer1_conv);
	mux_group.AddLayer(&layer2_maxpol);
	mux_group.AddLayer(&layer3_affine);
	mux_group.AddLayer(&layer4_affine);
	bb::NeuralNetBinaryMultiplex<float>	bin_mux(&mux_group, 28 * 28, 10, 1, 3);

	// build network
	bb::NeuralNet<> net;
	net.AddLayer(&bin_mux);

	// set optimizer
	bb::NeuralNetOptimizerAdam<> optimizerAdam;
	net.SetOptimizer(&optimizerAdam);

	// set binary mode
	net.SetBinaryMode(binary_mode);
	std::cout << "binary mode : " << binary_mode << std::endl;

	// set multiplexing size
	bin_mux.SetMuxSize(1);

	// run fitting
	bb::NeuralNetLossCrossEntropyWithSoftmax<>			lossFunc;
	bb::NeuralNetAccuracyCategoricalClassification<>	accFunc(num_class);
	net.Fitting(run_name, train_data, epoc_size, max_batch_size, &accFunc, &lossFunc, true, false);
}



void MnistSparseSimpleConvolutionBinToLut(int bin_epoc_size, size_t bin_max_batch_size, int lut_epoc_size, size_t lut_max_batch_size)
{
	// run name
	std::string run_name = "MnistSparseSimpleConvolutionBinToLut";
	int			num_class = 10;

	// load MNIST data
	auto train_data = bb::LoadMnist<>::Load();


	// -------- Binary-Network --------

	// sub network for layer0-convolution(3x3)
	bb::NeuralNetSparseBinaryAffine<>	bin_sub0_affine0(1 * 3 * 3, 96);
	bb::NeuralNetSparseBinaryAffine<>	bin_sub0_affine1(96, 16);
	bb::NeuralNetGroup<>				bin_sub0_net;
	bin_sub0_net.AddLayer(&bin_sub0_affine0);
	bin_sub0_net.AddLayer(&bin_sub0_affine1);

	// sub network for layer1-convolution(3x3)
	bb::NeuralNetSparseBinaryAffine<>	bin_sub1_affine0(16 * 3 * 3, 256);
	bb::NeuralNetSparseBinaryAffine<>	bin_sub1_affine1(256, 96);
	bb::NeuralNetSparseBinaryAffine<>	bin_sub1_affine2(96, 16);
	bb::NeuralNetGroup<>				bin_sub1_net;
	bin_sub1_net.AddLayer(&bin_sub1_affine0);
	bin_sub1_net.AddLayer(&bin_sub1_affine1);
	bin_sub1_net.AddLayer(&bin_sub1_affine2);

	// layers
	bb::NeuralNetConvolutionPack<>		bin_layer0_conv(&bin_sub0_net, 1, 28, 28, 16, 3, 3);
	bb::NeuralNetConvolutionPack<>		bin_layer1_conv(&bin_sub1_net, 16, 26, 26, 16, 3, 3);
	bb::NeuralNetMaxPooling<>			bin_layer2_maxpol(16, 24, 24, 2, 2);
	bb::NeuralNetSparseBinaryAffine<>	bin_layer3_affine(16 * 12 * 12, 180);
	bb::NeuralNetSparseBinaryAffine<>	bin_layer4_affine(180, 30);

	bb::NeuralNetGroup<>				bin_mux_group;
	bin_mux_group.AddLayer(&bin_layer0_conv);
	bin_mux_group.AddLayer(&bin_layer1_conv);
	bin_mux_group.AddLayer(&bin_layer2_maxpol);
	bin_mux_group.AddLayer(&bin_layer3_affine);
	bin_mux_group.AddLayer(&bin_layer4_affine);
	bb::NeuralNetBinaryMultiplex<float>	bin_mux(&bin_mux_group, 28 * 28, 10, 1, 3);

	// build network
	bb::NeuralNet<> bin_net;
	bin_net.AddLayer(&bin_mux);

	// set optimizer
	bb::NeuralNetOptimizerAdam<> optimizerAdam;
	bin_net.SetOptimizer(&optimizerAdam);

	// set binary mode
	bin_net.SetBinaryMode(true);

	// set multiplexing size
	bin_mux.SetMuxSize(1);

	// run fitting
	bb::NeuralNetLossCrossEntropyWithSoftmax<>			bin_lossFunc;
	bb::NeuralNetAccuracyCategoricalClassification<>	bin_accFunc(num_class);
	bin_net.Fitting(run_name + "_bin", train_data, bin_epoc_size, bin_max_batch_size, &bin_accFunc, &bin_lossFunc, true, true);



	// -------- LUT-Network --------

	// setup MNIST data
	auto& x_train = train_data.x_train;
	auto& y_train = train_data.y_train;
	auto& x_test = train_data.x_test;
	auto& y_test = train_data.y_test;
	auto label_train = bb::OnehotToLabel<std::uint8_t>(y_train);
	auto label_test = bb::OnehotToLabel<std::uint8_t>(y_test);
	auto train_size = x_train.size();
	auto test_size = x_test.size();
	auto x_node_size = x_test[0].size();


	std::mt19937_64 mt(1);

	// 学習時と評価時で多重化数(乱数を変えて複数枚通して集計できるようにする)を変える
	int lut_train_mux_size = 1;
	int lut_test_mux_size = 3;

	// Conv用subネット構築 (3x3)
	bb::NeuralNetBinaryLut6<true>		lut_sub0_lut0(1 * 3 * 3, 96);
	bb::NeuralNetBinaryLut6<true>		lut_sub0_lut1(96, 16);
	bb::NeuralNetGroup<>				lut_sub0_net;
	lut_sub0_net.AddLayer(&lut_sub0_lut0);
	lut_sub0_net.AddLayer(&lut_sub0_lut1);
	
	// Conv用subネット構築 (3x3)
	bb::NeuralNetBinaryLut6<true>		lut_sub1_lut0(16 * 3 * 3, 256);
	bb::NeuralNetBinaryLut6<true>		lut_sub1_lut1(256, 96);
	bb::NeuralNetBinaryLut6<true>		lut_sub1_lut2(96, 16);
	bb::NeuralNetGroup<>				lut_sub1_net;
	lut_sub1_net.AddLayer(&lut_sub1_lut0);
	lut_sub1_net.AddLayer(&lut_sub1_lut1);
	lut_sub1_net.AddLayer(&lut_sub1_lut2);

	bb::NeuralNetConvolutionPack<bool>	lut_layer0_conv(&lut_sub0_net, 1, 28, 28, 16, 3, 3);
	bb::NeuralNetConvolutionPack<bool>	lut_layer1_conv(&lut_sub1_net, 16, 26, 26, 16, 3, 3);
	bb::NeuralNetMaxPooling<bool>		lut_layer2_maxpol(16, 24, 24, 2, 2);
	bb::NeuralNetBinaryLut6<>			lut_layer3_lut(16 * 12 * 12, 180);
	bb::NeuralNetBinaryLut6<>			lut_layer4_lut(180, 30);

	bb::NeuralNetGroup<>				lut_mux_group;
	lut_mux_group.AddLayer(&lut_layer0_conv);
	lut_mux_group.AddLayer(&lut_layer1_conv);
	lut_mux_group.AddLayer(&lut_layer2_maxpol);
	lut_mux_group.AddLayer(&lut_layer3_lut);
	lut_mux_group.AddLayer(&lut_layer4_lut);
	bb::NeuralNetBinaryMultiplex<>		lut_mux(&lut_mux_group, 28 * 28, 10, 1, 3);

	// build network
	bb::NeuralNet<> lut_net;
	lut_net.AddLayer(&lut_mux);

	// set multiplexing size
	lut_mux.SetMuxSize(1);

	// copy
	std::cout << "[parameter copy] Binary-Neteork -> LUT-Network" << std::endl;
	lut_sub0_lut0.ImportLayer(bin_sub0_affine0);
	lut_sub0_lut1.ImportLayer(bin_sub0_affine1);
	lut_sub1_lut0.ImportLayer(bin_sub1_affine0);
	lut_sub1_lut1.ImportLayer(bin_sub1_affine1);
	lut_sub1_lut2.ImportLayer(bin_sub1_affine2);
	lut_layer3_lut.ImportLayer(bin_layer3_affine);
	lut_layer4_lut.ImportLayer(bin_layer4_affine);
	
	// 評価関数
	bb::NeuralNetAccuracyCategoricalClassification<>	lut_accFunc(num_class);

	// 初期評価
	lut_mux.SetMuxSize(lut_test_mux_size);	// 評価用の多重化数にスイッチ
	auto test_accuracy = lut_net.RunCalculation(train_data.x_test, train_data.y_test, lut_max_batch_size, 0, &lut_accFunc);
	std::cout << "initial test_accuracy : " << test_accuracy << std::endl;

	// start
	std::cout << "start : LUT-Network trainning" << std::endl;

	// 開始時間記録
	auto start_time = std::chrono::system_clock::now();

	// 学習ループ
	for (int epoc = 0; epoc < lut_epoc_size; ++epoc) {
		int iteration = 0;
		for (size_t train_index = 0; train_index < train_size; train_index += lut_max_batch_size) {
			// 末尾のバッチサイズクリップ
			size_t batch_size = std::min(lut_max_batch_size, train_size - train_index);
			if (batch_size < lut_max_batch_size) { break; }

			// 小サイズで演算すると劣化するので末尾スキップ
			if (batch_size < lut_max_batch_size) {
				break;
			}

			// バッチサイズ設定
			lut_mux.SetMuxSize(lut_train_mux_size);	// 学習の多重化数にスイッチ
			lut_net.SetBatchSize(batch_size);

			// データ格納
			auto in_sig_buf = lut_net.GetInputSignalBuffer();
			for (size_t frame = 0; frame < batch_size; ++frame) {
				for (size_t node = 0; node < x_node_size; ++node) {
					in_sig_buf.Set<float>(frame, node, x_train[train_index + frame][node]);
				}
			}

			// 予測
			lut_net.Forward(true);

			// バイナリ版フィードバック(力技学習)
			while (lut_mux.Feedback(lut_mux.GetOutputOnehotLoss<std::uint8_t, 10>(label_train, train_index)))
				;

			// 途中評価
			lut_mux.SetMuxSize(lut_test_mux_size);	// 評価用の多重化数にスイッチ
			auto test_accuracy = lut_net.RunCalculation(x_test, y_test, lut_max_batch_size, 0, &lut_accFunc);

			// 進捗表示
			auto progress = train_index + batch_size;
			auto rate = progress * 100 / train_size;
			std::cout << "[" << rate << "% (" << progress << "/" << train_size << ")]";
			std::cout << "  test_accuracy : " << test_accuracy << "                  ";
			std::cout << "\r" << std::flush;
		}

		// 評価
		lut_mux.SetMuxSize(lut_test_mux_size);	// 評価用の多重化数にスイッチ
		auto test_accuracy = lut_net.RunCalculation(x_test, y_test, lut_max_batch_size, 0, &lut_accFunc);
	//	auto train_accuracy = lut_net.RunCalculation(x_train, y_train, lut_max_batch_size, 0, &lut_accFunc);
		auto now_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time).count() / 1000.0;
		std::cout << now_time << "s " << "epoc[" << epoc + 1 << "]"
			<< "  test_accuracy : " << test_accuracy
	//		<< "  train_accuracy : " << train_accuracy
			<< std::endl;

		// Shuffle
		bb::ShuffleDataSet(mt(), x_train, y_train, label_train);
	}

	{
		// Write RTL
		std::ofstream ofs(run_name + "_lut_net.v");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, lut_sub0_lut0, "lutnet_conv0_layer0");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, lut_sub0_lut1, "lutnet_conv0_layer1");

		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, lut_sub1_lut0, "lutnet_conv1_layer0");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, lut_sub1_lut1, "lutnet_conv1_layer1");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, lut_sub1_lut2, "lutnet_conv1_layer2");

		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, lut_layer3_lut, "lutnet_layer3");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, lut_layer4_lut, "lutnet_layer4");
	}

	std::cout << "end\n" << std::endl;

}



void MnistSparseSimpleConvolutionLut(int epoc_size, size_t max_batch_size)
{
	// run name
	std::string run_name = "MnistSparseSimpleConvolutionLut";
	int			num_class = 10;

	// load MNIST data
	auto train_data = bb::LoadMnist<>::Load();
	auto& x_train = train_data.x_train;
	auto& y_train = train_data.y_train;
	auto& x_test = train_data.x_test;
	auto& y_test = train_data.y_test;
	auto label_train = bb::OnehotToLabel<std::uint8_t>(y_train);
	auto label_test = bb::OnehotToLabel<std::uint8_t>(y_test);
	auto train_size = x_train.size();
	auto test_size = x_test.size();
	auto x_node_size = x_test[0].size();

	std::mt19937_64 mt(1);

	// 学習時と評価時で多重化数(乱数を変えて複数枚通して集計できるようにする)を変える
	int lut_train_mux_size = 1;
	int lut_test_mux_size = 3;

	// Conv用subネット構築 (3x3)
	bb::NeuralNetBinaryLut6<true>	lut_sub0_lut0(1 * 3 * 3, 48);
	bb::NeuralNetBinaryLut6<true>		lut_sub0_lut1(48, 8);
	bb::NeuralNetGroup<>			lut_sub0_net;
	lut_sub0_net.AddLayer(&lut_sub0_lut0);
	lut_sub0_net.AddLayer(&lut_sub0_lut1);

	// Conv用subネット構築 (3x3)
	bb::NeuralNetBinaryLut6<true>	lut_sub1_lut0(8 * 3 * 3, 96);
	bb::NeuralNetBinaryLut6<true>	lut_sub1_lut1(96, 16);
	bb::NeuralNetGroup<>			lut_sub1_net;
	lut_sub1_net.AddLayer(&lut_sub1_lut0);
	lut_sub1_net.AddLayer(&lut_sub1_lut1);

	bb::NeuralNetConvolutionPack<bool>	lut_layer0_conv(&lut_sub0_net, 1, 28, 28, 8, 3, 3);
	bb::NeuralNetConvolutionPack<bool>	lut_layer1_conv(&lut_sub1_net, 8, 26, 26, 16, 3, 3);
	bb::NeuralNetMaxPooling<bool>		lut_layer2_maxpol(16, 24, 24, 2, 2);
	bb::NeuralNetBinaryLut6<>			lut_layer3_lut(16 * 12 * 12, 180);
	bb::NeuralNetBinaryLut6<>			lut_layer4_lut(180, 30);

	bb::NeuralNetGroup<>				lut_mux_group;
	lut_mux_group.AddLayer(&lut_layer0_conv);
	lut_mux_group.AddLayer(&lut_layer1_conv);
	lut_mux_group.AddLayer(&lut_layer2_maxpol);
	lut_mux_group.AddLayer(&lut_layer3_lut);
	lut_mux_group.AddLayer(&lut_layer4_lut);
	bb::NeuralNetBinaryMultiplex<>		lut_mux(&lut_mux_group, 28 * 28, 10, 1, 3);

	// build network
	bb::NeuralNet<> lut_net;
	lut_net.AddLayer(&lut_mux);

	// 評価関数
	bb::NeuralNetAccuracyCategoricalClassification<>	lut_accFunc(num_class);

	// 初期評価
	lut_mux.SetMuxSize(lut_test_mux_size);	// 評価用の多重化数にスイッチ
	auto test_accuracy = lut_net.RunCalculation(train_data.x_test, train_data.y_test, max_batch_size, 0, &lut_accFunc);
	std::cout << "initial test_accuracy : " << test_accuracy << std::endl;

	// start
	std::cout << "start : LUT-Network trainning" << std::endl;

	// 開始時間記録
	auto start_time = std::chrono::system_clock::now();

	// 学習ループ
	for (int epoc = 0; epoc < epoc_size; ++epoc) {
		int iteration = 0;
		for (size_t train_index = 0; train_index < train_size; train_index += max_batch_size) {
			// 末尾のバッチサイズクリップ
			size_t batch_size = std::min(max_batch_size, train_size - train_index);
			if (batch_size < max_batch_size) { break; }

			// 小サイズで演算すると劣化するので末尾スキップ
			if (batch_size < max_batch_size) {
				break;
			}

			// バッチサイズ設定
			lut_mux.SetMuxSize(lut_train_mux_size);	// 学習の多重化数にスイッチ
			lut_net.SetBatchSize(batch_size);

			// データ格納
			auto in_sig_buf = lut_net.GetInputSignalBuffer();
			for (size_t frame = 0; frame < batch_size; ++frame) {
				for (size_t node = 0; node < x_node_size; ++node) {
					in_sig_buf.Set<float>(frame, node, x_train[train_index + frame][node]);
				}
			}

			// 予測
			lut_net.Forward(true);

			// バイナリ版フィードバック(力技学習)
			while (lut_mux.Feedback(lut_mux.GetOutputOnehotLoss<std::uint8_t, 10>(label_train, train_index)))
				;

			// 途中評価
			lut_mux.SetMuxSize(lut_test_mux_size);	// 評価用の多重化数にスイッチ
			auto test_accuracy = lut_net.RunCalculation(x_test, y_test, max_batch_size, 0, &lut_accFunc);

			// 進捗表示
			auto progress = train_index + batch_size;
			auto rate = progress * 100 / train_size;
			std::cout << "[" << rate << "% (" << progress << "/" << train_size << ")]";
			std::cout << "  test_accuracy : " << test_accuracy << "                  ";
			std::cout << "\r" << std::flush;
		}

		// 評価
		lut_mux.SetMuxSize(lut_test_mux_size);	// 評価用の多重化数にスイッチ
		auto test_accuracy = lut_net.RunCalculation(x_test, y_test, max_batch_size, 0, &lut_accFunc);
		//	auto train_accuracy = lut_net.RunCalculation(x_train, y_train, lut_max_batch_size, 0, &lut_accFunc);
		auto now_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time).count() / 1000.0;
		std::cout << now_time << "s " << "epoc[" << epoc + 1 << "]"
			<< "  test_accuracy : " << test_accuracy
			//		<< "  train_accuracy : " << train_accuracy
			<< std::endl;

		// Shuffle
		bb::ShuffleDataSet(mt(), x_train, y_train, label_train);
	}

	{
		// Write RTL
		std::ofstream ofs(run_name + "_lut_net.v");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, lut_sub0_lut0, "lutnet_conv0_layer0");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, lut_sub0_lut1, "lutnet_conv0_layer1");

		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, lut_sub1_lut0, "lutnet_conv1_layer0");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, lut_sub1_lut1, "lutnet_conv1_layer1");

		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, lut_layer3_lut, "lutnet_layer3");
		bb::NeuralNetBinaryLut6VerilogXilinx(ofs, lut_layer4_lut, "lutnet_layer4");
	}

	std::cout << "end\n" << std::endl;

}

