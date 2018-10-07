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
#include "bb/NeuralNetSparseAffineBc.h"
#include "bb/NeuralNetSparseBinaryAffine.h"

#include "bb/NeuralNetRealToBinary.h"
#include "bb/NeuralNetBinaryToReal.h"
#include "bb/NeuralNetBinaryLut6.h"
#include "bb/NeuralNetBinaryLut6VerilogXilinx.h"
#include "bb/NeuralNetBinaryFilter.h"

#include "bb/NeuralNetSparseAffineSigmoid.h"

#include "bb/NeuralNetConvolutionPack.h"

#include "bb/NeuralNetOptimizerSgd.h"
#include "bb/NeuralNetOptimizerAdam.h"

#include "bb/NeuralNetConvolution.h"
#include "bb/NeuralNetMaxPooling.h"

#include "bb/ShuffleSet.h"

#include "mnist_read.h"



// MNISTデータを使った評価用クラス
class EvaluateMnist
{
protected:
	// 評価用データセット
	std::vector< std::vector<float> >	m_test_images;
	std::vector< std::uint8_t >			m_test_labels;
	std::vector< std::vector<float> >	m_test_onehot;

	// 学習用データセット
	std::vector< std::vector<float> >	m_train_images;
	std::vector< std::uint8_t >			m_train_labels;
	std::vector< std::vector<float> >	m_train_onehot;
	
	// 最大バッチサイズ
	size_t m_max_batch_size = 1024;

	// 乱数
	std::mt19937_64		m_mt;



public:
	// コンストラクタ
	EvaluateMnist(int train_max_size = -1, int test_max_size = -1)
	{
		// MNIST学習用データ読み込み
		m_train_images = mnist_read_images_real<float>("train-images-idx3-ubyte", train_max_size);
		m_train_labels = mnist_read_labels("train-labels-idx1-ubyte", train_max_size);

		// MNIST評価用データ読み込み
		m_test_images = mnist_read_images_real<float>("t10k-images-idx3-ubyte", test_max_size);
		m_test_labels = mnist_read_labels("t10k-labels-idx1-ubyte", test_max_size);

		// 元データがラベルなので、期待値だけ 1.0 となるクロスエントロピー用のデータも作る
		m_train_onehot = bb::LabelToOnehot<std::uint8_t, float>(m_train_labels, 10);
		m_test_onehot = bb::LabelToOnehot<std::uint8_t, float>(m_test_labels, 10);

		// 乱数初期化
		m_mt.seed(1);
	}


protected:
	// 学習データシャッフル
	void ShuffleTrainData(void)
	{
		bb::ShuffleDataSet(m_mt(), m_train_images, m_train_labels, m_train_onehot);
	}


	// 時間計測
	std::chrono::system_clock::time_point m_base_time;
	void reset_time(void) { m_base_time = std::chrono::system_clock::now(); }
	double get_time(void)
	{
		auto now_time = std::chrono::system_clock::now();
		return std::chrono::duration_cast<std::chrono::milliseconds>(now_time - m_base_time).count() / 1000.0;
	}

	// ネットの正解率評価
	double CalcAccuracy(bb::NeuralNet<>& net, std::vector< std::vector<float> >& images, std::vector<std::uint8_t>& labels)
	{
		int ok_count = 0;
		for (size_t x_index = 0; x_index < images.size(); x_index += m_max_batch_size) {
			// 末尾のバッチサイズクリップ
			size_t batch_size = std::min(m_max_batch_size, images.size() - x_index);

			// データセット
			net.SetBatchSize(batch_size);
			for (size_t frame = 0; frame < batch_size; ++frame) {
				net.SetInputSignal(frame, images[x_index + frame]);
			}

			// 評価実施
			net.Forward(false);

			// 結果集計
			for (size_t frame = 0; frame < batch_size; ++frame) {
				auto out_val = net.GetOutputSignal(frame);
				for (size_t i = 10; i < out_val.size(); i++) {
					out_val[i % 10] += out_val[i];
				}
				out_val.resize(10);
				int max_idx = bb::argmax<float>(out_val);
				ok_count += ((max_idx % 10) == (int)labels[x_index + frame] ? 1 : 0);
			}
		}

		// 正解率を返す
		return (double)ok_count / (double)images.size();
	}

	// 評価用データセットで正解率評価
	double CalcAccuracy(bb::NeuralNet<>& net)
	{
		return CalcAccuracy(net, m_test_images, m_test_labels);
	}


public:
#if 0
	// LUT6入力のバイナリ版の力技学習
	void RunBinaryLut6WithBbruteForce(int epoc_size, size_t max_batch_size, int max_iteration = -1)
	{
		std::cout << "start [RunBinaryLut6WithBbruteForce]" << std::endl;
		reset_time();

		std::mt19937_64 mt(1);

		// 学習時と評価時で多重化数(乱数を変えて複数毎通して集計できるようにする)を変える
		int train_mux_size = 1;
		int test_mux_size = 16;

		// 層構成定義
		size_t input_node_size = 28 * 28;
		size_t layer0_node_size = 360 * 2;
		size_t layer1_node_size = 60 * 4;
		size_t layer2_node_size = 10 * 8;
		size_t output_node_size = 10;

		// 学習用NET構築
		bb::NeuralNet<> net;
		bb::NeuralNetRealToBinary<> layer_real2bin(input_node_size, input_node_size, mt());
		bb::NeuralNetBinaryLut6<>	layer_lut0(input_node_size, layer0_node_size, mt());
		bb::NeuralNetBinaryLut6<>	layer_lut1(layer0_node_size, layer1_node_size, mt());
		bb::NeuralNetBinaryLut6<>	layer_lut2(layer1_node_size, layer2_node_size, mt());
		bb::NeuralNetBinaryToReal<>	layer_bin2real(layer2_node_size, output_node_size, mt());
		auto last_lut_layer = &layer_lut2;
		net.AddLayer(&layer_real2bin);
		net.AddLayer(&layer_lut0);
		net.AddLayer(&layer_lut1);
		net.AddLayer(&layer_lut2);
		//	net.AddLayer(&layer_unbinarize);	// 学習時はunbinarize不要

		// 評価用NET構築(ノードは共有)
		bb::NeuralNet<> net_eva;
		net_eva.AddLayer(&layer_real2bin);
		net_eva.AddLayer(&layer_lut0);
		net_eva.AddLayer(&layer_lut1);
		net_eva.AddLayer(&layer_lut2);
		net_eva.AddLayer(&layer_bin2real);

		// 学習ループ
		int iteration = 0;
		for (int epoc = 0; epoc < epoc_size; ++epoc) {
			// 学習状況評価
			layer_real2bin.InitializeCoeff(1);
			layer_bin2real.InitializeCoeff(1);
			net_eva.SetMuxSize(test_mux_size);
			std::cout << get_time() << "s " << "epoc[" << epoc << "] accuracy : " << CalcAccuracy(net_eva) << std::endl;

			for (size_t x_index = 0; x_index < m_train_images.size(); x_index += max_batch_size) {
				// 末尾のバッチサイズクリップ
				size_t batch_size = std::min(max_batch_size, m_train_images.size() - x_index);

				// バッチ学習データの作成
				std::vector< std::vector<float> >	batch_images(m_train_images.begin() + x_index, m_train_images.begin() + x_index + batch_size);
				std::vector< std::uint8_t >			batch_labels(m_train_labels.begin() + x_index, m_train_labels.begin() + x_index + batch_size);

				// データセット
				net.SetMuxSize(train_mux_size);
				net.SetBatchSize(batch_size);
				for (size_t frame = 0; frame < batch_size; ++frame) {
					net.SetInputSignal(frame, batch_images[frame]);
				}

				// 予測
				net.Forward();

				// バイナリ版フィードバック(力技学習)
				while (net.Feedback(last_lut_layer->GetOutputOnehotLoss<std::uint8_t, 10>(batch_labels)))
					;

				// 中間表示()
				layer_real2bin.InitializeCoeff(1);
				layer_bin2real.InitializeCoeff(1);
				net_eva.SetMuxSize(test_mux_size);
				std::cout << get_time() << "s " << "epoc[" << epoc << "] accuracy : " << CalcAccuracy(net_eva) << std::endl;

				iteration++;
				if (max_iteration > 0 && iteration >= max_iteration) {
					goto loop_end;
				}
			}
		}
	loop_end:


		{
			std::ofstream ofs("lut_net.v");
			bb::NeuralNetBinaryLut6VerilogXilinx(ofs, layer_lut0, "lutnet_layer0");
			bb::NeuralNetBinaryLut6VerilogXilinx(ofs, layer_lut1, "lutnet_layer1");
			bb::NeuralNetBinaryLut6VerilogXilinx(ofs, layer_lut2, "lutnet_layer2");
		}

		std::cout << "end\n" << std::endl;
	}
#endif

	// LUT6入力のバイナリ版の力技学習
	void RunBinaryLut6WithBbruteForce(int epoc_size, size_t max_batch_size, int max_iteration = -1)
	{
		std::cout << "start [RunBinaryLut6WithBbruteForce]" << std::endl;
		reset_time();

		std::mt19937_64 mt(1);

		// 学習時と評価時で多重化数(乱数を変えて複数毎通して集計できるようにする)を変える
		int train_mux_size = 1;
		int test_mux_size = 16;

		// 層構成定義
		size_t input_node_size = 28 * 28;
		size_t output_node_size = 10;
		size_t input_hmux_size = 1;
		size_t output_hmux_size = 8;

		size_t layer0_node_size = 360 * 2;
		size_t layer1_node_size = 60 * 4;
		size_t layer2_node_size = output_node_size * output_hmux_size;

		// バイナリネットのGroup作成
		bb::NeuralNetBinaryLut6<>	bin_layer0_lut(input_node_size*input_hmux_size, layer0_node_size);
		bb::NeuralNetBinaryLut6<>	bin_layer1_lut(layer0_node_size, layer1_node_size);
		bb::NeuralNetBinaryLut6<>	bin_layer2_lut(layer1_node_size, layer2_node_size);
		bb::NeuralNetGroup<>		bin_group;
		bin_group.AddLayer(&bin_layer0_lut);
		bin_group.AddLayer(&bin_layer1_lut);
		bin_group.AddLayer(&bin_layer2_lut);
		auto last_lut_layer = &bin_layer2_lut;

		// 多重化してパッキング
		bb::NeuralNetBinaryMultiplex<>	bin_mux(&bin_group, input_node_size, output_node_size, input_hmux_size, output_hmux_size);

		// ネット構築
		bb::NeuralNet<> net;
		net.AddLayer(&bin_mux);


		// 学習ループ
		int iteration = 0;
		for (int epoc = 0; epoc < epoc_size; ++epoc) {
			// 学習状況評価
			bin_mux.SetMuxSize(test_mux_size);	// 評価用の多重化数にスイッチ
			std::cout << get_time() << "s " << "epoc[" << epoc << "] accuracy : " << CalcAccuracy(net) << std::endl;

			for (size_t x_index = 0; x_index < m_train_images.size(); x_index += max_batch_size) {
				// 末尾のバッチサイズクリップ
				size_t batch_size = std::min(max_batch_size, m_train_images.size() - x_index);

				// バッチ学習データの作成
				std::vector< std::vector<float> >	batch_images(m_train_images.begin() + x_index, m_train_images.begin() + x_index + batch_size);
				std::vector< std::uint8_t >			batch_labels(m_train_labels.begin() + x_index, m_train_labels.begin() + x_index + batch_size);

				// データセット
				bin_mux.SetMuxSize(train_mux_size);
				net.SetBatchSize(batch_size);
				for (size_t frame = 0; frame < batch_size; ++frame) {
					net.SetInputSignal(frame, batch_images[frame]);
				}

				// 予測
				net.Forward();

				// バイナリ版フィードバック(力技学習)
				while (bin_mux.Feedback(bin_mux.GetOutputOnehotLoss<std::uint8_t, 10>(batch_labels)))
					;

				// 中間表示()
				bin_mux.SetMuxSize(test_mux_size);
				std::cout << get_time() << "s " << "epoc[" << epoc << "] accuracy : " << CalcAccuracy(net) << std::endl;

				iteration++;
				if (max_iteration > 0 && iteration >= max_iteration) {
					goto loop_end;
				}
			}
		}
	loop_end:

		{
			std::ofstream ofs("lut_net.v");
			bb::NeuralNetBinaryLut6VerilogXilinx(ofs, bin_layer0_lut, "lutnet_layer0");
			bb::NeuralNetBinaryLut6VerilogXilinx(ofs, bin_layer1_lut, "lutnet_layer1");
			bb::NeuralNetBinaryLut6VerilogXilinx(ofs, bin_layer2_lut, "lutnet_layer2");
		}
		
		std::cout << "end\n" << std::endl;
	}



	// 実数(float)の全接続層で、フラットなネットを評価
	void RunDenseAffineSigmoid(int epoc_size, size_t max_batch_size, double learning_rate)
	{
		std::cout << "start [RunDenseAffineSigmoid]" << std::endl;
		reset_time();

		// 実数版NET構築
		bb::NeuralNet<> net;
		bb::NeuralNetAffine<>  layer0_affine(28 * 28, 100);
		bb::NeuralNetSigmoid<> layer0_sigmoid(100);
		bb::NeuralNetAffine<>  layer1_affine(100, 10);
		bb::NeuralNetSoftmax<> layer1_softmax(10);
		net.AddLayer(&layer0_affine);
		net.AddLayer(&layer0_sigmoid);
		net.AddLayer(&layer1_affine);
		net.AddLayer(&layer1_softmax);

		for (int epoc = 0; epoc < epoc_size; ++epoc) {

			// 学習状況評価
			std::cout << get_time() << "s " << "epoc[" << epoc << "] accuracy : " << CalcAccuracy(net) << std::endl;


			for (size_t x_index = 0; x_index < m_train_images.size(); x_index += max_batch_size) {
				// 末尾のバッチサイズクリップ
				size_t batch_size = std::min(max_batch_size, m_train_images.size() - x_index);

				// データセット
				net.SetBatchSize(batch_size);
				for (size_t frame = 0; frame < batch_size; ++frame) {
					net.SetInputSignal(frame, m_train_images[x_index + frame]);
				}

				// 予測
				net.Forward();

				// 誤差逆伝播
				for (size_t frame = 0; frame < batch_size; ++frame) {
					auto signals = net.GetOutputSignal(frame);
					for (size_t node = 0; node < signals.size(); ++node) {
						signals[node] -= m_train_onehot[x_index + frame][node];
						signals[node] /= (float)batch_size;
					}
					net.SetOutputError(frame, signals);
				}
				net.Backward();

				// 更新
				net.Update();
			}
		}
		std::cout << "end\n" << std::endl;
	}


	void RunDenseAffineReLU(int epoc_size, size_t max_batch_size, double learning_rate)
	{
		std::cout << "start [RunDenseAffineReLU]" << std::endl;
		reset_time();

		// 実数版NET構築
		bb::NeuralNet<> net;
		bb::NeuralNetAffine<>  layer0_affine(28 * 28, 100);
		bb::NeuralNetReLU<>	   layer0_relu(100);
		bb::NeuralNetAffine<>  layer1_affine(100, 10);
		bb::NeuralNetSoftmax<> layer1_softmax(10);
		net.AddLayer(&layer0_affine);
		net.AddLayer(&layer0_relu);
		net.AddLayer(&layer1_affine);
		net.AddLayer(&layer1_softmax);

		for (int epoc = 0; epoc < epoc_size; ++epoc) {

			// 学習状況評価
			std::cout << get_time() << "s " << "epoc[" << epoc << "] accuracy : " << CalcAccuracy(net) << std::endl;


			for (size_t x_index = 0; x_index < m_train_images.size(); x_index += max_batch_size) {
				// 末尾のバッチサイズクリップ
				size_t batch_size = std::min(max_batch_size, m_train_images.size() - x_index);

				// データセット
				net.SetBatchSize(batch_size);
				for (size_t frame = 0; frame < batch_size; ++frame) {
					net.SetInputSignal(frame, m_train_images[x_index + frame]);
				}

				// 予測
				net.Forward();

				// 誤差逆伝播
				for (size_t frame = 0; frame < batch_size; ++frame) {
					auto signals = net.GetOutputSignal(frame);
					for (size_t node = 0; node < signals.size(); ++node) {
						signals[node] -= m_train_onehot[x_index + frame][node];
						signals[node] /= (float)batch_size;
					}
					net.SetOutputError(frame, signals);
				}
				net.Backward();

				// 更新
				net.Update();
			}
		}
		std::cout << "end\n" << std::endl;
	}


	// 実数(float)の全接続層で、フラットなネットを評価
	void RunDenseAffineBatchNorm(int epoc_size, size_t max_batch_size, double learning_rate)
	{
		std::cout << "start [RunDenseAffineBatchNorm]" << std::endl;
		reset_time();

		// 実数版NET構築
		bb::NeuralNet<> net;
		bb::NeuralNetAffine<>				layer0_affine(28 * 28, 100);
		bb::NeuralNetBatchNormalization<>	layer0_batch_norm(100);
		bb::NeuralNetSigmoid<>				layer0_sigmoid(100);
		bb::NeuralNetAffine<>				layer1_affine(100, 10);
		bb::NeuralNetBatchNormalization<>	layer1_batch_norm(10);
		bb::NeuralNetSoftmax<>				layer1_softmax(10);
		net.AddLayer(&layer0_affine);
		net.AddLayer(&layer0_batch_norm);
		net.AddLayer(&layer0_sigmoid);
		net.AddLayer(&layer1_affine);
		net.AddLayer(&layer1_batch_norm);
		net.AddLayer(&layer1_softmax);

		for (int epoc = 0; epoc < epoc_size; ++epoc) {

			// 学習状況評価
			std::cout << get_time() << "s " << "epoc[" << epoc << "] accuracy : " << CalcAccuracy(net) << std::endl;


			for (size_t x_index = 0; x_index < m_train_images.size(); x_index += max_batch_size) {
				// 末尾のバッチサイズクリップ
				size_t batch_size = std::min(max_batch_size, m_train_images.size() - x_index);

				// データセット
				net.SetBatchSize(batch_size);
				for (size_t frame = 0; frame < batch_size; ++frame) {
					net.SetInputSignal(frame, m_train_images[x_index + frame]);
				}

				// 予測
				net.Forward();

				// 誤差逆伝播
				for (size_t frame = 0; frame < batch_size; ++frame) {
					auto signals = net.GetOutputSignal(frame);
					for (size_t node = 0; node < signals.size(); ++node) {
						signals[node] -= m_train_onehot[x_index + frame][node];
						signals[node] /= (float)batch_size;
					}
					net.SetOutputError(frame, signals);
				}
				net.Backward();

				// 更新
				net.Update();
			}
		}
		std::cout << "end\n" << std::endl;
	}


	// 実数(float)で6入力に制限ノードで層を形成して、フラットなネットを評価
	void RunSparseAffineSigmoid(int epoc_size, size_t max_batch_size)
	{
		std::cout << "start [RunSparseAffineSigmoid]" << std::endl;
		reset_time();

		// 実数版NET構築
		bb::NeuralNet<> net;
		size_t input_node_size = 28 * 28;
		size_t layer0_node_size = 10 * 6 * 6 * 3;
		size_t layer1_node_size = 10 * 6 * 6;
		size_t layer2_node_size = 10 * 6;
		size_t output_node_size = 10;
		bb::NeuralNetRealToBinary<float>	input_real2bin(28 * 28, 28 * 28);
		bb::NeuralNetSparseAffineSigmoid<>	layer0_affine(28 * 28, layer0_node_size);
		bb::NeuralNetSparseAffineSigmoid<>	layer1_affine(layer0_node_size, layer1_node_size);
		bb::NeuralNetSparseAffineSigmoid<>	layer2_affine(layer1_node_size, layer2_node_size);
		bb::NeuralNetSparseAffineSigmoid<>	layer3_affine(layer2_node_size, output_node_size);
		bb::NeuralNetBinaryToReal<float>	output_bin2real(output_node_size, output_node_size);
		bb::NeuralNetSoftmax<>				output_softmax(output_node_size);
		net.AddLayer(&input_real2bin);
		net.AddLayer(&layer0_affine);
		net.AddLayer(&layer1_affine);
		net.AddLayer(&layer2_affine);
		net.AddLayer(&layer3_affine);
		net.AddLayer(&output_bin2real);
		net.AddLayer(&output_softmax);

		net.SetOptimizer(&bb::NeuralNetOptimizerAdam<>());

		net.SetBinaryMode(true);

		for (int epoc = 0; epoc < epoc_size; ++epoc) {

			// 学習状況評価
			std::cout << get_time() << "s " << "epoc[" << epoc << "] accuracy : " << CalcAccuracy(net) << std::endl;
			
			for (size_t x_index = 0; x_index < m_train_images.size(); x_index += max_batch_size) {
				// 末尾のバッチサイズクリップ
				size_t batch_size = std::min(max_batch_size, m_train_images.size() - x_index);

				// データセット
				net.SetBatchSize(batch_size);
				for (size_t frame = 0; frame < batch_size; ++frame) {
					net.SetInputSignal(frame, m_train_images[x_index + frame]);
				}

				// 予測
				net.Forward();

				// 誤差逆伝播
				for (size_t frame = 0; frame < batch_size; ++frame) {
					auto signals = net.GetOutputSignal(frame);
					for (size_t node = 0; node < signals.size(); ++node) {
						signals[node] -= m_train_onehot[x_index + frame][node];
						signals[node] /= (float)batch_size;
					}
					net.SetOutputError(frame, signals);
				}
				net.Backward();
				
				// 更新
				net.Update();
			}
		}
		std::cout << "end\n" << std::endl;
	}


	// 浮動小数点で学習させてバイナリにコピー
	void RunRealToBinary(int epoc_size, size_t max_batch_size)
	{
		std::cout << "start [RunRealToBinary]" << std::endl;
		reset_time();

		std::mt19937_64 mt(1);

		int train_mux_size = 1;
		int test_mux_size = 16;


		// 層構成
		size_t input_node_size = 28 * 28;
		size_t layer0_node_size = 10 * 6 * 6 * 3;
		size_t layer1_node_size = 10 * 6 * 6;
		size_t layer2_node_size = 10 * 6;
		size_t output_node_size = 10;

		// 実数版NET構築
		bb::NeuralNetSparseAffineSigmoid<6>	real_mux0_affine(input_node_size, layer0_node_size);
		bb::NeuralNetSparseAffineSigmoid<6>	real_mux1_affine(layer0_node_size, layer1_node_size);
		bb::NeuralNetSparseAffineSigmoid<6>	real_mux2_affine(layer1_node_size, layer2_node_size);
		bb::NeuralNetSparseAffineSigmoid<6>	real_mux3_affine(layer2_node_size, output_node_size);
		bb::NeuralNetGroup<>				real_mux_group;
		real_mux_group.AddLayer(&real_mux0_affine);
		real_mux_group.AddLayer(&real_mux1_affine);
		real_mux_group.AddLayer(&real_mux2_affine);
		real_mux_group.AddLayer(&real_mux3_affine);

		bb::NeuralNetBinaryMultiplex<float>	real_mux(&real_mux_group, input_node_size, output_node_size, 1, 1);
		bb::NeuralNetSoftmax<>				real_softmax(output_node_size);
		bb::NeuralNet<> real_net;
		real_net.AddLayer(&real_mux);
		real_net.AddLayer(&real_softmax);

		real_mux.SetMuxSize(1);

		real_net.SetOptimizer(&bb::NeuralNetOptimizerAdam<>());

		real_net.SetBinaryMode(true);


		// 実数で逆伝播で学習
		for (int epoc = 0; epoc < epoc_size; ++epoc) {

			// 学習状況評価
	//		real_mux.SetMuxSize(test_mux_size);
			auto real_accuracy = CalcAccuracy(real_net);
			std::cout << get_time() << "s " << "epoc[" << epoc << "] real_net accuracy : " << real_accuracy << std::endl;

			for (size_t x_index = 0; x_index < m_train_images.size(); x_index += max_batch_size) {
				// 末尾のバッチサイズクリップ
				size_t batch_size = std::min(max_batch_size, m_train_images.size() - x_index);

				// 入力データ設定
	//			real_mux.SetMuxSize(train_mux_size);
				real_net.SetBatchSize(batch_size);
				for (size_t frame = 0; frame < batch_size; ++frame) {
					real_net.SetInputSignal(frame, m_train_images[frame + x_index]);
				}

				// 予測
				real_net.Forward();

				// 誤差逆伝播
				for (size_t frame = 0; frame < batch_size; ++frame) {
					auto signals = real_net.GetOutputSignal(frame);
					for (size_t node = 0; node < signals.size(); ++node) {
						signals[node] -= m_train_onehot[frame + x_index][node % 10];
						signals[node] /= (float)batch_size;
					}
					real_net.SetOutputError(frame, signals);
				}
				real_net.Backward();

				// 更新
				real_net.Update();
			}
		}

		// 最終結果表示
		std::cout << get_time() << "s " << "real_net final accuracy : " << CalcAccuracy(real_net) << std::endl;


		// バイナリ版NET構築
		bb::NeuralNetBinaryLut6<>	bin_mux0_lut(input_node_size, layer0_node_size);
		bb::NeuralNetBinaryLut6<>	bin_mux1_lut(layer0_node_size, layer1_node_size);
		bb::NeuralNetBinaryLut6<>	bin_mux2_lut(layer1_node_size, layer2_node_size);
		bb::NeuralNetBinaryLut6<>	bin_mux3_lut(layer2_node_size, output_node_size);
		bb::NeuralNetGroup<>		bin_mux_group;
		bin_mux_group.AddLayer(&bin_mux0_lut);
		bin_mux_group.AddLayer(&bin_mux1_lut);
		bin_mux_group.AddLayer(&bin_mux2_lut);
		bin_mux_group.AddLayer(&bin_mux3_lut);
		bb::NeuralNetBinaryMultiplex<>	bin_mux(&bin_mux_group, input_node_size, output_node_size, 1, 1);

		bb::NeuralNet<> bin_net;
		bin_net.AddLayer(&bin_mux);


		// パラメータをコピー
		std::cout << "[parameter copy] real-net -> binary-net" << std::endl;
		bin_mux0_lut.ImportLayer(real_mux0_affine);
		bin_mux1_lut.ImportLayer(real_mux1_affine);
		bin_mux2_lut.ImportLayer(real_mux2_affine);
		bin_mux3_lut.ImportLayer(real_mux3_affine);

		// バイナリ版評価
		bin_mux.SetMuxSize(test_mux_size);

		// 学習ループ
		max_batch_size = 8192;
		int max_iteration = 8;
		int iteration = 0;
		for (int epoc = 0; epoc < epoc_size; ++epoc) {
			// 学習状況評価
			bin_mux.SetMuxSize(test_mux_size);
			std::cout << get_time() << "s " << "epoc[" << epoc << "] bin_net accuracy : " << CalcAccuracy(bin_net) << std::endl;

			for (size_t x_index = 0; x_index < m_train_images.size(); x_index += max_batch_size) {
				// 末尾のバッチサイズクリップ
				size_t batch_size = std::min(max_batch_size, m_train_images.size() - x_index);

				// バッチ学習データの作成
				std::vector< std::vector<float> >	batch_images(m_train_images.begin() + x_index, m_train_images.begin() + x_index + batch_size);
				std::vector< std::uint8_t >			batch_labels(m_train_labels.begin() + x_index, m_train_labels.begin() + x_index + batch_size);

				// データセット
				bin_mux.SetMuxSize(train_mux_size);
				bin_net.SetBatchSize(batch_size);
				for (size_t frame = 0; frame < batch_size; ++frame) {
					bin_net.SetInputSignal(frame, batch_images[frame]);
				}

				// 予測
				bin_net.Forward();

				// バイナリ版フィードバック(力技学習)
				while (bin_mux.Feedback(bin_mux.GetOutputOnehotLoss<std::uint8_t, 10>(batch_labels)))
					;

				// 中間表示
				bin_mux.SetMuxSize(test_mux_size);
				std::cout << get_time() << "s " << "epoc[" << epoc << "] bin_net accuracy : " << CalcAccuracy(bin_net) << std::endl;

				iteration++;
				if (max_iteration > 0 && iteration >= max_iteration) {
					goto loop_end;
				}
			}
		}
	loop_end:

		{
			std::ofstream ofs("lut_net.v");
			bb::NeuralNetBinaryLut6VerilogXilinx(ofs, bin_mux0_lut, "lutnet_layer0");
			bb::NeuralNetBinaryLut6VerilogXilinx(ofs, bin_mux1_lut, "lutnet_layer1");
			bb::NeuralNetBinaryLut6VerilogXilinx(ofs, bin_mux2_lut, "lutnet_layer2");
		}

		std::cout << "end\n" << std::endl;
	}



	////////////////////////////
	// ここから畳み込み
	////////////////////////////


	// 実数(float)の畳み込み確認
	void RunConvolutionReal(int epoc_size, size_t max_batch_size, double learning_rate)
	{
		std::cout << "start [RunConvolutionReal]" << std::endl;
		reset_time();
		
		// 実数版NET構築
		bb::NeuralNet<> net;
		bb::NeuralNetConvolution<>  layer0_conv(1, 28, 28, 16, 5, 5);
		bb::NeuralNetSigmoid<>		layer0_sigmoid(24*24*16);
		bb::NeuralNetAffine<>		layer1_affine(24 * 24 * 16, 10);
		bb::NeuralNetSoftmax<>		layer1_softmax(10);
		net.AddLayer(&layer0_conv);
		net.AddLayer(&layer0_sigmoid);
		net.AddLayer(&layer1_affine);
		net.AddLayer(&layer1_softmax);

		for (int epoc = 0; epoc < epoc_size; ++epoc) {

			// 学習状況評価
			std::cout << get_time() << "s " << "epoc[" << epoc << "] accuracy : " << CalcAccuracy(net) << std::endl;


			for (size_t x_index = 0; x_index < m_train_images.size(); x_index += max_batch_size) {
				// 末尾のバッチサイズクリップ
				size_t batch_size = std::min(max_batch_size, m_train_images.size() - x_index);

				// データセット
				net.SetBatchSize(batch_size);
				for (size_t frame = 0; frame < batch_size; ++frame) {
					net.SetInputSignal(frame, m_train_images[x_index + frame]);
				}

				// 予測
				net.Forward();

				// 誤差逆伝播
				for (size_t frame = 0; frame < batch_size; ++frame) {
					auto signals = net.GetOutputSignal(frame);
					for (size_t node = 0; node < signals.size(); ++node) {
						signals[node] -= m_train_onehot[x_index + frame][node];
						signals[node] /= (float)batch_size;
					}
					net.SetOutputError(frame, signals);
				}
				net.Backward();

				// 更新
				net.Update();
			}
		}
		std::cout << "end\n" << std::endl;
	}


	// 実数(float)の畳み込み確認
	void RunCnnRealPre(int epoc_size, size_t max_batch_size, double learning_rate)
	{
		std::cout << "start [RunCnnRealPre]" << std::endl;
		reset_time();

		// 実数版NET構築
		bb::NeuralNet<> net;
		bb::NeuralNetConvolution<>  layer0_conv(1, 28, 28, 30, 5, 5);
		bb::NeuralNetReLU<>			layer0_relu(24 * 24 * 30);
		bb::NeuralNetMaxPooling<>	layer0_maxpol(30, 24, 24, 2, 2);
		bb::NeuralNetAffine<>		layer1_affine(30 * 12 * 12, 100);
		bb::NeuralNetReLU<>			layer1_reru(100);
		bb::NeuralNetAffine<>		layer2_affine(100, 10);
		bb::NeuralNetSoftmax<>		layer2_softmax(10);
		net.AddLayer(&layer0_conv);
		net.AddLayer(&layer0_relu);
		net.AddLayer(&layer0_maxpol);
		net.AddLayer(&layer1_affine);
		net.AddLayer(&layer1_reru);
		net.AddLayer(&layer2_affine);
		net.AddLayer(&layer2_softmax);

		for (int epoc = 0; epoc < epoc_size; ++epoc) {

			// 学習状況評価
			std::cout << get_time() << "s " << "epoc[" << epoc << "] accuracy : " << CalcAccuracy(net) << std::endl;


			for (size_t x_index = 0; x_index < m_train_images.size(); x_index += max_batch_size) {
				// 末尾のバッチサイズクリップ
				size_t batch_size = std::min(max_batch_size, m_train_images.size() - x_index);

				// データセット
				net.SetBatchSize(batch_size);
				for (size_t frame = 0; frame < batch_size; ++frame) {
					net.SetInputSignal(frame, m_train_images[x_index + frame]);
				}

				// 予測
				net.Forward();

				// 誤差逆伝播
				for (size_t frame = 0; frame < batch_size; ++frame) {
					auto signals = net.GetOutputSignal(frame);
					for (size_t node = 0; node < signals.size(); ++node) {
						signals[node] -= m_train_onehot[x_index + frame][node];
						signals[node] /= (float)batch_size;
					}
					net.SetOutputError(frame, signals);
				}
				net.Backward();

				// 更新
				net.Update();
			}
		}
		std::cout << "end\n" << std::endl;
	}


	// 「ゼロから作る」の構成
	void RunSimpleConv(int epoc_size, size_t max_batch_size, double learning_rate)
	{
		std::cout << "start [RunSimpleConv]" << std::endl;
		reset_time();

		// 実数版NET構築
		bb::NeuralNet<> net;
		bb::NeuralNetConvolution<>  layer0_conv(1, 28, 28, 30, 5, 5);
		bb::NeuralNetReLU<>			layer0_relu(24 * 24 * 30);
		bb::NeuralNetMaxPooling<>	layer0_maxpol(30, 24, 24, 2, 2);

		bb::NeuralNetAffine<>		layer1_affine(30*12*12, 100);
		bb::NeuralNetReLU<>			layer1_relu(100);

		bb::NeuralNetAffine<>		layer2_affine(100, 10);
		bb::NeuralNetSoftmax<>		layer2_softmax(10);
		net.AddLayer(&layer0_conv);
		net.AddLayer(&layer0_relu);
		net.AddLayer(&layer0_maxpol);
		net.AddLayer(&layer1_affine);
		net.AddLayer(&layer1_relu);
		net.AddLayer(&layer2_affine);
		net.AddLayer(&layer2_softmax);

		for (int epoc = 0; epoc < epoc_size; ++epoc) {

			// 学習状況評価
			std::cout << get_time() << "s " << "epoc[" << epoc << "] accuracy : " << CalcAccuracy(net) << std::endl;


			for (size_t x_index = 0; x_index < m_train_images.size(); x_index += max_batch_size) {
				// 末尾のバッチサイズクリップ
				size_t batch_size = std::min(max_batch_size, m_train_images.size() - x_index);

				// データセット
				net.SetBatchSize(batch_size);
				for (size_t frame = 0; frame < batch_size; ++frame) {
					net.SetInputSignal(frame, m_train_images[x_index + frame]);
				}

				// 予測
				net.Forward();

				// 誤差逆伝播
				for (size_t frame = 0; frame < batch_size; ++frame) {
					auto signals = net.GetOutputSignal(frame);
					for (size_t node = 0; node < signals.size(); ++node) {
						signals[node] -= m_train_onehot[x_index + frame][node];
						signals[node] /= (float)batch_size;
					}
					net.SetOutputError(frame, signals);
				}
				net.Backward();

				// 更新
				net.Update();
			}
		}
		std::cout << "end\n" << std::endl;
	}


	void RunSimpleConvReLU(int epoc_size, size_t max_batch_size)
	{
		std::cout << "start [RunSimpleConvReLU]" << std::endl;
		reset_time();

		// 実数版NET構築
		bb::NeuralNet<> net;
		bb::NeuralNetConvolution<>  layer0_conv(1, 28, 28, 16, 3, 3);
		bb::NeuralNetReLU<>			layer0_relu(16 * 26 * 26);
		
		bb::NeuralNetConvolution<>  layer1_conv(16, 26, 26, 16, 3, 3);
		bb::NeuralNetReLU<>			layer1_relu(16 * 24 * 24);

		bb::NeuralNetMaxPooling<>	layer2_maxpol(16, 24, 24, 2, 2);

		bb::NeuralNetAffine<>		layer3_affine(16 * 12 * 12, 10);
		bb::NeuralNetSoftmax<>		layer3_softmax(10);
		net.AddLayer(&layer0_conv);
		net.AddLayer(&layer0_relu);
		net.AddLayer(&layer1_conv);
		net.AddLayer(&layer1_relu);
		net.AddLayer(&layer2_maxpol);
		net.AddLayer(&layer3_affine);
		net.AddLayer(&layer3_softmax);


		layer0_conv.SetOptimizer(&bb::NeuralNetOptimizerAdam<>(0.001f, 0.9f, 0.999f));
		layer1_conv.SetOptimizer(&bb::NeuralNetOptimizerAdam<>(0.001f, 0.9f, 0.999f));
		layer3_affine.SetOptimizer(&bb::NeuralNetOptimizerAdam<>(0.001f, 0.9f, 0.999f));


		for (int epoc = 0; epoc < epoc_size; ++epoc) {

			// 学習状況評価
			std::cout << get_time() << "s " << "epoc[" << epoc << "] accuracy : " << CalcAccuracy(net) << std::endl;


			for (size_t x_index = 0; x_index < m_train_images.size(); x_index += max_batch_size) {
				// 末尾のバッチサイズクリップ
				size_t batch_size = std::min(max_batch_size, m_train_images.size() - x_index);

				// データセット
				net.SetBatchSize(batch_size);
				for (size_t frame = 0; frame < batch_size; ++frame) {
					net.SetInputSignal(frame, m_train_images[x_index + frame]);
				}

				// 予測
				net.Forward();

				// 誤差逆伝播
				for (size_t frame = 0; frame < batch_size; ++frame) {
					auto signals = net.GetOutputSignal(frame);
					for (size_t node = 0; node < signals.size(); ++node) {
						signals[node] -= m_train_onehot[x_index + frame][node];
						signals[node] /= (float)batch_size;
					}
					net.SetOutputError(frame, signals);
				}
				net.Backward();

				// 更新
				net.Update();
			}
		}
		std::cout << "end\n" << std::endl;
	}


	// 実験比較用
	void RunSimpleConvSigmoid(int epoc_size, size_t max_batch_size)
	{
		std::cout << "start [RunSimpleConvSigmoid]" << std::endl;
		reset_time();

		// 実数版NET構築
		bb::NeuralNet<> net;
		bb::NeuralNetConvolution<>  layer0_conv(1, 28, 28, 16, 3, 3);
		bb::NeuralNetSigmoid<>		layer0_sigmoid(16 * 26 * 26);

		bb::NeuralNetConvolution<>  layer1_conv(16, 26, 26, 16, 3, 3);
		bb::NeuralNetSigmoid<>		layer1_sigmoid(16 * 24 * 24);

		bb::NeuralNetMaxPooling<>	layer2_maxpol(16, 24, 24, 2, 2);

		bb::NeuralNetAffine<>		layer3_affine(16 * 12 * 12, 10);
		bb::NeuralNetSoftmax<>		layer3_softmax(10);
		net.AddLayer(&layer0_conv);
		net.AddLayer(&layer0_sigmoid);
		net.AddLayer(&layer1_conv);
		net.AddLayer(&layer1_sigmoid);
		net.AddLayer(&layer2_maxpol);
		net.AddLayer(&layer3_affine);
		net.AddLayer(&layer3_softmax);

		net.SetOptimizer(&bb::NeuralNetOptimizerAdam<>(0.001f, 0.9f, 0.999f));

//		layer0_conv.SetOptimizer(&bb::NeuralNetOptimizerAdam<>(0.001f, 0.9f, 0.999f));
//		layer1_conv.SetOptimizer(&bb::NeuralNetOptimizerAdam<>(0.001f, 0.9f, 0.999f));
//		layer3_affine.SetOptimizer(&bb::NeuralNetOptimizerAdam<>(0.001f, 0.9f, 0.999f));
		

		for (int epoc = 0; epoc < epoc_size; ++epoc) {

			// 学習状況評価
			std::cout << get_time() << "s " << "epoc[" << epoc << "] accuracy : " << CalcAccuracy(net) << std::endl;


			for (size_t x_index = 0; x_index < m_train_images.size(); x_index += max_batch_size) {
				// 末尾のバッチサイズクリップ
				size_t batch_size = std::min(max_batch_size, m_train_images.size() - x_index);

				// データセット
				net.SetBatchSize(batch_size);
				for (size_t frame = 0; frame < batch_size; ++frame) {
					net.SetInputSignal(frame, m_train_images[x_index + frame]);
				}

				// 予測
				net.Forward();

				// 誤差逆伝播
				for (size_t frame = 0; frame < batch_size; ++frame) {
					auto signals = net.GetOutputSignal(frame);
					for (size_t node = 0; node < signals.size(); ++node) {
						signals[node] -= m_train_onehot[x_index + frame][node];
						signals[node] /= (float)batch_size;
					}
					net.SetOutputError(frame, signals);
				}
				net.Backward();

				// 更新
				net.Update();
			}
		}
		std::cout << "end\n" << std::endl;
	}


	void RunSimpleConvBinary(int epoc_size, size_t max_batch_size)
	{
		std::cout << "start [RunSimpleConvBinary]" << std::endl;
		reset_time();

		// 実数版NET構築
//		bb::NeuralNetRealToBinary<float>	layer0_rel2bin(28*28, 28 * 28);
		bb::NeuralNetConvolution<>			layer0_conv(1, 28, 28, 16, 3, 3);
		bb::NeuralNetBatchNormalization<>	layer0_norm(16 * 26 * 26);
		bb::NeuralNetSigmoid<>				layer0_activate(16 * 26 * 26);

		bb::NeuralNetConvolution<>			layer1_conv(16, 26, 26, 16, 3, 3);
		bb::NeuralNetBatchNormalization<>	layer1_norm(16 * 24 * 24);
		bb::NeuralNetSigmoid<>				layer1_activate(16 * 24 * 24);

		bb::NeuralNetMaxPooling<>			layer2_maxpol(16, 24, 24, 2, 2);

		bb::NeuralNetAffine<>				layer3_affine(16 * 12 * 12, 30);
		bb::NeuralNetBatchNormalization<>	layer3_norm(30);
		bb::NeuralNetSigmoid<>				layer3_activate(30);

		bb::NeuralNetBinaryToReal<float>	layer4_bin2rel(30, 10);
//		bb::NeuralNetSigmoid<>				layer4_activate(10);
		bb::NeuralNetSoftmax<>				layer4_softmax(10);


		bb::NeuralNet<> net;
//		net.AddLayer(&layer0_rel2bin);
		net.AddLayer(&layer0_conv);
		net.AddLayer(&layer0_norm);
		net.AddLayer(&layer0_activate);
		net.AddLayer(&layer1_conv);
		net.AddLayer(&layer1_norm);
		net.AddLayer(&layer1_activate);
		net.AddLayer(&layer2_maxpol);
		net.AddLayer(&layer3_affine);
		net.AddLayer(&layer3_norm);
		net.AddLayer(&layer3_activate);
		net.AddLayer(&layer4_bin2rel);
//		net.AddLayer(&layer4_activate);
		net.AddLayer(&layer4_softmax);

//		layer0_activate.SetBinaryMode(true);
//		layer1_activate.SetBinaryMode(true);
//		layer3_activate.SetBinaryMode(true);
//		layer4_activate.SetBinaryMode(true);


		net.SetOptimizer(&bb::NeuralNetOptimizerAdam<>(0.001f, 0.9f, 0.999f));

//		layer0_conv.SetOptimizer(&bb::NeuralNetOptimizerAdam<>(0.001, 0.9, 0.999));
//		layer0_norm.SetOptimizer(&bb::NeuralNetOptimizerAdam<>(0.001, 0.9, 0.999));
//		layer1_conv.SetOptimizer(&bb::NeuralNetOptimizerAdam<>(0.001, 0.9, 0.999));
//		layer1_norm.SetOptimizer(&bb::NeuralNetOptimizerAdam<>(0.001, 0.9, 0.999));
//		layer3_affine.SetOptimizer(&bb::NeuralNetOptimizerAdam<>(0.001, 0.9, 0.999));


		for (int epoc = 0; epoc < epoc_size; ++epoc) {

			// 学習状況評価
			auto accuracy = CalcAccuracy(net);
			std::cout << get_time() << "s " << "epoc[" << epoc << "] accuracy : " << accuracy << std::endl;

			if ( accuracy > 0.7 ) {
				std::cout << " [binary mode] : enable" << std::endl;
				layer0_activate.SetBinaryMode(true);
				layer1_activate.SetBinaryMode(true);
				layer3_activate.SetBinaryMode(true);
			//	layer4_activate.SetBinaryMode(true);
			}

			for (size_t x_index = 0; x_index < m_train_images.size(); x_index += max_batch_size) {
				// 末尾のバッチサイズクリップ
				size_t batch_size = std::min(max_batch_size, m_train_images.size() - x_index);

				// データセット
				net.SetBatchSize(batch_size);
				for (size_t frame = 0; frame < batch_size; ++frame) {
					net.SetInputSignal(frame, m_train_images[x_index + frame]);
				}

				// 予測
				net.Forward();

				// 誤差逆伝播
				for (size_t frame = 0; frame < batch_size; ++frame) {
					auto signals = net.GetOutputSignal(frame);
					for (size_t node = 0; node < signals.size(); ++node) {
						signals[node] -= m_train_onehot[x_index + frame][node];
						signals[node] /= (float)batch_size;
					}
					net.SetOutputError(frame, signals);
				}
				net.Backward();

				// 更新
				net.Update();
			}
		}
		std::cout << "end\n" << std::endl;
	}

	void RunFullyConvBinary(int epoc_size, size_t max_batch_size)
	{
		std::cout << "start [RunFullyConvBinary]" << std::endl;
		reset_time();

		// 実数版NET構築
		//		bb::NeuralNetRealToBinary<float>	layer0_rel2bin(28*28, 28 * 28);
		bb::NeuralNetConvolution<>			layer0_conv(1, 28, 28, 30, 3, 3);
		bb::NeuralNetBatchNormalization<>	layer0_norm(30 * 26 * 26);
		bb::NeuralNetSigmoid<>				layer0_activate(30 * 26 * 26);

		bb::NeuralNetConvolution<>			layer1_conv(30, 26, 26, 30, 3, 3);
		bb::NeuralNetBatchNormalization<>	layer1_norm(30 * 24 * 24);
		bb::NeuralNetSigmoid<>				layer1_activate(30 * 24 * 24);

		bb::NeuralNetMaxPooling<>			layer2_maxpol(30, 24, 24, 2, 2);

		bb::NeuralNetConvolution<>			layer3_conv(30, 12, 12, 30, 3, 3);
		bb::NeuralNetBatchNormalization<>	layer3_norm(30 * 10 * 10);
		bb::NeuralNetSigmoid<>				layer3_activate(30 * 10 * 10);

		bb::NeuralNetConvolution<>			layer4_conv(30, 10, 10, 30, 3, 3);
		bb::NeuralNetBatchNormalization<>	layer4_norm(30 * 8 * 8);
		bb::NeuralNetSigmoid<>				layer4_activate(30 * 8 * 8);

		bb::NeuralNetMaxPooling<>			layer5_maxpol(30, 8, 8, 2, 2);

		bb::NeuralNetConvolution<>			layer6_conv(30, 4, 4, 30, 3, 3);
		bb::NeuralNetBatchNormalization<>	layer6_norm(30 * 2 * 2);
		bb::NeuralNetSigmoid<>				layer6_activate(30 * 2 * 2);

		bb::NeuralNetConvolution<>			layer7_conv(30, 2, 2, 30, 2, 2);
		bb::NeuralNetBatchNormalization<>	layer7_norm(30 * 1 * 1);
		bb::NeuralNetSigmoid<>				layer7_activate(30 * 1 * 1);

		bb::NeuralNetBinaryToReal<float>	layer8_bin2rel(30, 10);
		//		bb::NeuralNetSigmoid<>				layer4_activate(10);
		bb::NeuralNetSoftmax<>				layer8_softmax(10);


		bb::NeuralNet<> net;
		//		net.AddLayer(&layer0_rel2bin);
		net.AddLayer(&layer0_conv);
		net.AddLayer(&layer0_norm);
		net.AddLayer(&layer0_activate);
		net.AddLayer(&layer1_conv);
		net.AddLayer(&layer1_norm);
		net.AddLayer(&layer1_activate);
		net.AddLayer(&layer2_maxpol);

		net.AddLayer(&layer3_conv);
		net.AddLayer(&layer3_norm);
		net.AddLayer(&layer3_activate);
		net.AddLayer(&layer4_conv);
		net.AddLayer(&layer4_norm);
		net.AddLayer(&layer4_activate);
		net.AddLayer(&layer5_maxpol);

		net.AddLayer(&layer6_conv);
		net.AddLayer(&layer6_norm);
		net.AddLayer(&layer6_activate);
		net.AddLayer(&layer7_conv);
		net.AddLayer(&layer7_norm);
		net.AddLayer(&layer7_activate);

		net.AddLayer(&layer8_bin2rel);
		//		net.AddLayer(&layer8_activate);
		net.AddLayer(&layer8_softmax);

		//		layer0_activate.SetBinaryMode(true);
		//		layer1_activate.SetBinaryMode(true);
		//		layer3_activate.SetBinaryMode(true);
		//		layer4_activate.SetBinaryMode(true);

		layer0_activate.SetBinaryMode(true);
		layer1_activate.SetBinaryMode(true);
		layer3_activate.SetBinaryMode(true);
		layer4_activate.SetBinaryMode(true);
		layer6_activate.SetBinaryMode(true);
		layer7_activate.SetBinaryMode(true);

		net.SetOptimizer(&bb::NeuralNetOptimizerAdam<>(0.001f, 0.9f, 0.999f));

		net.SetBinaryMode(true);


		//		layer0_conv.SetOptimizer(&bb::NeuralNetOptimizerAdam<>(0.001, 0.9, 0.999));
		//		layer0_norm.SetOptimizer(&bb::NeuralNetOptimizerAdam<>(0.001, 0.9, 0.999));
		//		layer1_conv.SetOptimizer(&bb::NeuralNetOptimizerAdam<>(0.001, 0.9, 0.999));
		//		layer1_norm.SetOptimizer(&bb::NeuralNetOptimizerAdam<>(0.001, 0.9, 0.999));
		//		layer3_affine.SetOptimizer(&bb::NeuralNetOptimizerAdam<>(0.001, 0.9, 0.999));


		for (int epoc = 0; epoc < epoc_size; ++epoc) {

			// 学習状況評価
			auto accuracy = CalcAccuracy(net);
			std::cout << get_time() << "s " << "epoc[" << epoc << "] accuracy : " << accuracy << std::endl;

			if (accuracy > 0.9) {
				std::cout << " [binary mode] : enable" << std::endl;
				net.SetBinaryMode(true);
				layer0_activate.SetBinaryMode(true);
				layer1_activate.SetBinaryMode(true);
				layer3_activate.SetBinaryMode(true);
				layer4_activate.SetBinaryMode(true);
				layer6_activate.SetBinaryMode(true);
				layer7_activate.SetBinaryMode(true);

				accuracy = CalcAccuracy(net);
				std::cout << get_time() << "s " << "epoc[" << epoc << "] accuracy : " << accuracy << std::endl;
			}

			for (size_t x_index = 0; x_index < m_train_images.size(); x_index += max_batch_size) {
				// 末尾のバッチサイズクリップ
				size_t batch_size = std::min(max_batch_size, m_train_images.size() - x_index);

				// データセット
				net.SetBatchSize(batch_size);
				for (size_t frame = 0; frame < batch_size; ++frame) {
					net.SetInputSignal(frame, m_train_images[x_index + frame]);
				}

				// 予測
				net.Forward();

				// 誤差逆伝播
				for (size_t frame = 0; frame < batch_size; ++frame) {
					auto signals = net.GetOutputSignal(frame);
					for (size_t node = 0; node < signals.size(); ++node) {
						signals[node] -= m_train_onehot[x_index + frame][node];
						signals[node] /= (float)batch_size;
					}
					net.SetOutputError(frame, signals);
				}
				net.Backward();

				// 更新
				net.Update();
			}
		}
		std::cout << "end\n" << std::endl;
	}


	// 本命
	void RunSparseFullyCnn(int epoc_size, size_t max_batch_size)
	{
		std::ofstream ofs_log("log.txt");

		ofs_log   << "start [RunSparseFullyCnn]" << std::endl;
		std::cout << "start [RunSparseFullyCnn]" << std::endl;

		reset_time();

		// Conv用subネット構築 (3x3)
		bb::NeuralNetSparseAffineSigmoid<>	sub0_affine(1 * 3 * 3, 30);
		bb::NeuralNetGroup<>				sub0_net;
		sub0_net.AddLayer(&sub0_affine);

		// Conv用subネット構築 (3x3)
		bb::NeuralNetSparseAffineSigmoid<>	sub1_affine0(30 * 3 * 3, 180);
		bb::NeuralNetSparseAffineSigmoid<>	sub1_affine1(180, 30);
		bb::NeuralNetGroup<>				sub1_net;
		sub1_net.AddLayer(&sub1_affine0);
		sub1_net.AddLayer(&sub1_affine1);

		// Conv用subネット構築 (3x3)
		bb::NeuralNetSparseAffineSigmoid<>	sub3_affine0(30 * 3 * 3, 180);
		bb::NeuralNetSparseAffineSigmoid<>	sub3_affine1(180, 30);
		bb::NeuralNetGroup<>				sub3_net;
		sub3_net.AddLayer(&sub3_affine0);
		sub3_net.AddLayer(&sub3_affine1);

		// Conv用subネット構築 (3x3)
		bb::NeuralNetSparseAffineSigmoid<>	sub4_affine0(30 * 3 * 3, 180);
		bb::NeuralNetSparseAffineSigmoid<>	sub4_affine1(180, 30);
		bb::NeuralNetGroup<>				sub4_net;
		sub4_net.AddLayer(&sub4_affine0);
		sub4_net.AddLayer(&sub4_affine1);

		// Conv用subネット構築 (3x3)
		bb::NeuralNetSparseAffineSigmoid<>	sub6_affine0(30 * 3 * 3, 180);
		bb::NeuralNetSparseAffineSigmoid<>	sub6_affine1(180, 30);
		bb::NeuralNetGroup<>				sub6_net;
		sub6_net.AddLayer(&sub6_affine0);
		sub6_net.AddLayer(&sub6_affine1);

		// Conv用subネット構築 (3x3)
		bb::NeuralNetSparseAffineSigmoid<>	sub7_affine0(30 * 2 * 2, 180);
		bb::NeuralNetSparseAffineSigmoid<>	sub7_affine1(180, 30);
		bb::NeuralNetGroup<>				sub7_net;
		sub7_net.AddLayer(&sub7_affine0);
		sub7_net.AddLayer(&sub7_affine1);
		
		// 実数版NET構築
		bb::NeuralNet<> net;
		bb::NeuralNetConvolutionPack<>		layer0_conv(&sub0_net, 1, 28, 28, 30, 3, 3);
		bb::NeuralNetConvolutionPack<>		layer1_conv(&sub1_net, 30, 26, 26, 30, 3, 3);
		bb::NeuralNetMaxPooling<>			layer2_maxpol(30, 24, 24, 2, 2);
		bb::NeuralNetConvolutionPack<>		layer3_conv(&sub3_net, 30, 12, 12, 30, 3, 3);
		bb::NeuralNetConvolutionPack<>		layer4_conv(&sub4_net, 30, 10, 10, 30, 3, 3);
		bb::NeuralNetMaxPooling<>			layer5_maxpol(30, 8, 8, 2, 2);
		bb::NeuralNetConvolutionPack<>		layer6_conv(&sub6_net, 30, 4, 4, 30, 3, 3);
		bb::NeuralNetConvolutionPack<>		layer7_conv(&sub7_net, 30, 2, 2, 30, 2, 2);
		bb::NeuralNetBinaryToReal<float>	layer8_bin2rel(30, 10);
		bb::NeuralNetSoftmax<>				layer9_softmax(10);

		net.AddLayer(&layer0_conv);
		net.AddLayer(&layer1_conv);
		net.AddLayer(&layer2_maxpol);
		net.AddLayer(&layer3_conv);
		net.AddLayer(&layer4_conv);
		net.AddLayer(&layer5_maxpol);
		net.AddLayer(&layer6_conv);
		net.AddLayer(&layer7_conv);
		net.AddLayer(&layer8_bin2rel);
		net.AddLayer(&layer9_softmax);

		net.SetOptimizer(&bb::NeuralNetOptimizerAdam<>(0.001f, 0.9f, 0.999f));
		net.SetBinaryMode(true);

		for (int epoc = 0; epoc < epoc_size; ++epoc) {

			// 学習状況評価
			auto accuracy = CalcAccuracy(net);
			ofs_log << get_time() << "s " << "epoc[" << epoc << "] accuracy : " << accuracy << std::endl;
			std::cout << get_time() << "s " << "epoc[" << epoc << "] accuracy : " << accuracy << std::endl;

			for (size_t x_index = 0; x_index < m_train_images.size(); x_index += max_batch_size) {
				// 末尾のバッチサイズクリップ
				size_t batch_size = std::min(max_batch_size, m_train_images.size() - x_index);

				// データセット
				net.SetBatchSize(batch_size);
				for (size_t frame = 0; frame < batch_size; ++frame) {
					net.SetInputSignal(frame, m_train_images[x_index + frame]);
				}

				// 予測
				net.Forward();

				// 誤差逆伝播
				for (size_t frame = 0; frame < batch_size; ++frame) {
					auto signals = net.GetOutputSignal(frame);
					for (size_t node = 0; node < signals.size(); ++node) {
						signals[node] -= m_train_onehot[x_index + frame][node];
						signals[node] /= (float)batch_size;
					}
					net.SetOutputError(frame, signals);
				}
				net.Backward();

				// 更新
				net.Update();
			}
		}
		ofs_log << "end\n" << std::endl;
		std::cout << "end\n" << std::endl;
	}
	

	// 実験比較用
	void RunSimpleConvPackSigmoid(int epoc_size, size_t max_batch_size)
	{
		std::cout << "start [RunSimpleConvPackSigmoid]" << std::endl;
		reset_time();

		// Conv用subネット構築 (3x3)
		bb::NeuralNetSparseBinaryAffine<>	sub0_affine(1 * 3 * 3, 16);
		bb::NeuralNetGroup<>				sub0_net;
		sub0_net.AddLayer(&sub0_affine);

		// Conv用subネット構築 (3x3)
		bb::NeuralNetSparseBinaryAffine<>	sub1_affine(16 * 3 * 3, 16);
		bb::NeuralNetGroup<>				sub1_net;
		sub1_net.AddLayer(&sub1_affine);


		// 実数版NET構築
		bb::NeuralNet<> net;
		bb::NeuralNetConvolutionPack<>	layer0_conv(&sub0_net, 1, 28, 28, 16, 3, 3);
		bb::NeuralNetSigmoid<>			layer0_sigmoid(16 * 26 * 26);

		bb::NeuralNetConvolutionPack<>	layer1_conv(&sub1_net, 16, 26, 26, 16, 3, 3);
		bb::NeuralNetSigmoid<>			layer1_sigmoid(16 * 24 * 24);

		bb::NeuralNetMaxPooling<>		layer2_maxpol(16, 24, 24, 2, 2);

		bb::NeuralNetAffine<>			layer3_affine(16 * 12 * 12, 10);
		bb::NeuralNetSoftmax<>			layer3_softmax(10);
		net.AddLayer(&layer0_conv);
		net.AddLayer(&layer0_sigmoid);
		net.AddLayer(&layer1_conv);
		net.AddLayer(&layer1_sigmoid);
		net.AddLayer(&layer2_maxpol);
		net.AddLayer(&layer3_affine);
		net.AddLayer(&layer3_softmax);
		
		net.SetOptimizer(&bb::NeuralNetOptimizerAdam<>(0.001f, 0.9f, 0.999f));
		net.SetBinaryMode(true);

		for (int epoc = 0; epoc < epoc_size; ++epoc) {

			// 学習状況評価
			std::cout << get_time() << "s " << "epoc[" << epoc << "] accuracy : " << CalcAccuracy(net) << std::endl;


			for (size_t x_index = 0; x_index < m_train_images.size(); x_index += max_batch_size) {
				// 末尾のバッチサイズクリップ
				size_t batch_size = std::min(max_batch_size, m_train_images.size() - x_index);

				// データセット
				net.SetBatchSize(batch_size);
				for (size_t frame = 0; frame < batch_size; ++frame) {
					net.SetInputSignal(frame, m_train_images[x_index + frame]);
				}

				// 予測
				net.Forward();

				// 誤差逆伝播
				for (size_t frame = 0; frame < batch_size; ++frame) {
					auto signals = net.GetOutputSignal(frame);
					for (size_t node = 0; node < signals.size(); ++node) {
						signals[node] -= m_train_onehot[x_index + frame][node];
						signals[node] /= (float)batch_size;
					}
					net.SetOutputError(frame, signals);
				}
				net.Backward();

				// 更新
				net.Update();
			}
		}
		std::cout << "end\n" << std::endl;
	}


	// Fully-CNN
	void RunFullyCnnReal(int epoc_size, size_t max_batch_size, double learning_rate)
	{
		std::cout << "start [RunFullyCnnReal]" << std::endl;
		reset_time();

		// 実数版NET構築
		size_t layer0_c_size = 32;
		size_t layer1_c_size = 16;
		size_t layer2_c_size = 10;
		bb::NeuralNet<> net;
		bb::NeuralNetConvolution<>  layer0_conv(1, 28, 28, layer0_c_size, 5, 5);
		bb::NeuralNetSigmoid<>		layer0_relu(24 * 24 * layer0_c_size);
		bb::NeuralNetMaxPooling<>	layer0_maxpol(layer0_c_size, 24, 24, 2, 2);

		bb::NeuralNetConvolution<>  layer1_conv(layer0_c_size, 12, 12, layer1_c_size, 5, 5);
		bb::NeuralNetSigmoid<>		layer1_relu(8 * 8 * layer1_c_size);
		bb::NeuralNetMaxPooling<>	layer1_maxpol(layer1_c_size, 8, 8, 2, 2);

		bb::NeuralNetConvolution<>  layer2_conv(layer1_c_size, 4, 4, layer2_c_size, 3, 3);
		bb::NeuralNetSigmoid<>		layer2_relu(2 * 2 * layer2_c_size);
		bb::NeuralNetMaxPooling<>	layer2_maxpol(layer2_c_size, 2, 2, 2, 2);

		bb::NeuralNetSoftmax<>		layer3_softmax(layer2_c_size);
		net.AddLayer(&layer0_conv);
		net.AddLayer(&layer0_relu);
		net.AddLayer(&layer0_maxpol);
		net.AddLayer(&layer1_conv);
		net.AddLayer(&layer1_relu);
		net.AddLayer(&layer1_maxpol);
		net.AddLayer(&layer2_conv);
		net.AddLayer(&layer2_relu);
		net.AddLayer(&layer2_maxpol);
		net.AddLayer(&layer3_softmax);

		for (int epoc = 0; epoc < epoc_size; ++epoc) {

			// 学習状況評価
			std::cout << get_time() << "s " << "epoc[" << epoc << "] accuracy : " << CalcAccuracy(net) << std::endl;


			for (size_t x_index = 0; x_index < m_train_images.size(); x_index += max_batch_size) {
				// 末尾のバッチサイズクリップ
				size_t batch_size = std::min(max_batch_size, m_train_images.size() - x_index);

				// データセット
				net.SetBatchSize(batch_size);
				for (size_t frame = 0; frame < batch_size; ++frame) {
					net.SetInputSignal(frame, m_train_images[x_index + frame]);
				}

				// 予測
				net.Forward();

				// 誤差逆伝播
				for (size_t frame = 0; frame < batch_size; ++frame) {
					auto signals = net.GetOutputSignal(frame);
					for (size_t node = 0; node < signals.size(); ++node) {
						signals[node] -= m_train_onehot[x_index + frame][node];
						signals[node] /= (float)batch_size;
					}
					net.SetOutputError(frame, signals);
				}
				net.Backward();

				// 更新
				net.Update();
			}
		}
		std::cout << "end\n" << std::endl;
	}


	// binary
	void RunSimpleConvBinary(int epoc_size, size_t max_batch_size, double learning_rate)
	{
		std::cout << "start [RunSimpleConvBinary]" << std::endl;
		reset_time();

		// Conv用subネット構築 (1x5x5 -> 8)
		bb::NeuralNetSparseBinaryAffine<>	sub_affine0(5 * 5, 32);
		bb::NeuralNetSparseBinaryAffine<>	sub_affine1(32, 8);
		bb::NeuralNetGroup<>				sub_net;
		sub_net.AddLayer(&sub_affine0);
		sub_net.AddLayer(&sub_affine1);


		// 実数版NET構築
		size_t layer0_node_size = 8 * 12 * 12;
		size_t layer1_node_size = 360 * 1;
		size_t layer2_node_size = 60 * 2;
		size_t layer3_node_size = 10 * 2;
		size_t output_node_size = 10;
		bb::NeuralNetRealToBinary<float>	layer0_real2bin(28 * 28, 28 * 28);
		bb::NeuralNetConvolutionPack<>		layer0_conv(&sub_net, 1, 28, 28, 8, 5, 5);
		bb::NeuralNetMaxPooling<>			layer0_maxpol(8, 24, 24, 2, 2);
		bb::NeuralNetSparseBinaryAffine<>	layer1_affine(layer0_node_size, layer1_node_size);
		bb::NeuralNetSparseBinaryAffine<>	layer2_affine(layer1_node_size, layer2_node_size);
		bb::NeuralNetSparseBinaryAffine<>	layer3_affine(layer2_node_size, layer3_node_size);
		bb::NeuralNetBinaryToReal<float>	layer3_bin2real(layer3_node_size, output_node_size);

		bb::NeuralNet<> net;
		net.AddLayer(&layer0_real2bin);
		net.AddLayer(&layer0_conv);
		net.AddLayer(&layer0_maxpol);
		net.AddLayer(&layer1_affine);
		net.AddLayer(&layer2_affine);
		net.AddLayer(&layer3_affine);
		net.AddLayer(&layer3_bin2real);

		for (int epoc = 0; epoc < epoc_size; ++epoc) {

			// 学習状況評価
			std::cout << get_time() << "s " << "epoc[" << epoc << "] accuracy : " << CalcAccuracy(net) << std::endl;

			for (size_t x_index = 0; x_index < m_train_images.size(); x_index += max_batch_size) {
				// 末尾のバッチサイズクリップ
				size_t batch_size = std::min(max_batch_size, m_train_images.size() - x_index);

				// データセット
				net.SetBatchSize(batch_size);
				for (size_t frame = 0; frame < batch_size; ++frame) {
					net.SetInputSignal(frame, m_train_images[x_index + frame]);
				}

				// 予測
				net.Forward();

				// 誤差逆伝播
				for (size_t frame = 0; frame < batch_size; ++frame) {
					auto signals = net.GetOutputSignal(frame);
					for (size_t node = 0; node < signals.size(); ++node) {
						signals[node] -= m_train_onehot[x_index + frame][node];
						signals[node] /= (float)batch_size;
					}
					net.SetOutputError(frame, signals);
				}
				net.Backward();

				// 更新
				net.Update();
			}
		}
		std::cout << "end\n" << std::endl;
	}

	// binary fully
	void RunFullyCnnBinary(int epoc_size, size_t max_batch_size, double learning_rate)
	{
		std::cout << "start [RunFullyCnnBinary]" << std::endl;
		reset_time();

		size_t layer0_c_size = 32;
		size_t layer1_c_size = 60;
		size_t layer2_c_size = 30;

		// Conv用subネット構築 (5x5)
		bb::NeuralNetSparseBinaryAffine<>	sub0_affine0(1 * 5 * 5, 128);
		bb::NeuralNetSparseBinaryAffine<>	sub0_affine1(128, layer0_c_size);
		bb::NeuralNetGroup<>				sub0_net;
		sub0_net.AddLayer(&sub0_affine0);
		sub0_net.AddLayer(&sub0_affine1);

		// Conv用subネット構築 (5x5)
		bb::NeuralNetSparseBinaryAffine<>	sub1_affine0(layer0_c_size * 5 * 5, 256);
		bb::NeuralNetSparseBinaryAffine<>	sub1_affine1(256, layer1_c_size);
		bb::NeuralNetGroup<>				sub1_net;
		sub1_net.AddLayer(&sub1_affine0);
		sub1_net.AddLayer(&sub1_affine1);

		// Conv用subネット構築 (3x3)
		bb::NeuralNetSparseBinaryAffine<>	sub2_affine0(layer1_c_size * 3 * 3, 128);
		bb::NeuralNetSparseBinaryAffine<>	sub2_affine1(128, layer2_c_size);
		bb::NeuralNetGroup<>				sub2_net;
		sub2_net.AddLayer(&sub2_affine0);
		sub2_net.AddLayer(&sub2_affine1);


		// 実数版NET構築
		bb::NeuralNetRealToBinary<float>	layer0_real2bin(28 * 28, 28 * 28);
		bb::NeuralNetConvolutionPack<>		layer0_conv(&sub0_net, 1, 28, 28, layer0_c_size, 5, 5);
		bb::NeuralNetMaxPooling<>			layer0_maxpol(layer0_c_size, 24, 24, 2, 2);

		bb::NeuralNetConvolutionPack<>		layer1_conv(&sub1_net, layer0_c_size, 12, 12, layer1_c_size, 5, 5);
		bb::NeuralNetMaxPooling<>			layer1_maxpol(layer1_c_size, 8, 8, 2, 2);

		bb::NeuralNetConvolutionPack<>		layer2_conv(&sub2_net, layer1_c_size, 4, 4, layer2_c_size, 3, 3);
		bb::NeuralNetMaxPooling<>			layer2_maxpol(layer2_c_size, 2, 2, 2, 2);
		bb::NeuralNetRealToBinary<float>	layer2_real2bin(layer2_c_size, 10);
		bb::NeuralNetSoftmax<>				layer2_softmax(10);

		bb::NeuralNet<> net;
		net.AddLayer(&layer0_real2bin);
		net.AddLayer(&layer0_conv);
		net.AddLayer(&layer0_maxpol);
		net.AddLayer(&layer1_conv);
		net.AddLayer(&layer1_maxpol);
		net.AddLayer(&layer2_conv);
		net.AddLayer(&layer2_maxpol);
		net.AddLayer(&layer2_real2bin);
		net.AddLayer(&layer2_softmax);

		for (int epoc = 0; epoc < epoc_size; ++epoc) {

			// 学習状況評価
			std::cout << get_time() << "s " << "epoc[" << epoc << "] accuracy : " << CalcAccuracy(net) << std::endl;

			for (size_t x_index = 0; x_index < m_train_images.size(); x_index += max_batch_size) {
				// 末尾のバッチサイズクリップ
				size_t batch_size = std::min(max_batch_size, m_train_images.size() - x_index);

				// データセット
				net.SetBatchSize(batch_size);
				for (size_t frame = 0; frame < batch_size; ++frame) {
					net.SetInputSignal(frame, m_train_images[x_index + frame]);
				}

				// 予測
				net.Forward();

				// 誤差逆伝播
				for (size_t frame = 0; frame < batch_size; ++frame) {
					auto signals = net.GetOutputSignal(frame);
					for (size_t node = 0; node < signals.size(); ++node) {
						signals[node] -= m_train_onehot[x_index + frame][node];
						signals[node] /= (float)batch_size;
					}
					net.SetOutputError(frame, signals);
				}
				net.Backward();

				// 更新
				net.Update();
			}
		}
		std::cout << "end\n" << std::endl;
	}


	// binary cnv 実験
	void RunSimpleConvBinary2(int epoc_size, size_t max_batch_size, double learning_rate)
	{
		std::cout << "start [RunSimpleConvBinary2]" << std::endl;
		reset_time();

		// Conv用subネット構築 (3x3)
		bb::NeuralNetSparseBinaryAffine<>	sub0_affine0(1 * 3 * 3, 96);
		bb::NeuralNetSparseBinaryAffine<>	sub0_affine1(96, 96);
		bb::NeuralNetSparseBinaryAffine<>	sub0_affine2(96, 16);
		bb::NeuralNetGroup<>				sub0_net;
		sub0_net.AddLayer(&sub0_affine0);
		sub0_net.AddLayer(&sub0_affine1);
		sub0_net.AddLayer(&sub0_affine2);

		// Conv用subネット構築 (3x3)
		bb::NeuralNetSparseBinaryAffine<>	sub1_affine0(16 * 3 * 3, 192);
		bb::NeuralNetSparseBinaryAffine<>	sub1_affine1(192, 96);
		bb::NeuralNetSparseBinaryAffine<>	sub1_affine2(96, 16);
		bb::NeuralNetGroup<>				sub1_net;
		sub1_net.AddLayer(&sub1_affine0);
		sub1_net.AddLayer(&sub1_affine1);
		sub1_net.AddLayer(&sub1_affine2);

		// 実数版NET構築
		bb::NeuralNetRealToBinary<float>	layer0_real2bin(28 * 28, 28 * 28);
		bb::NeuralNetConvolutionPack<>		layer0_conv(&sub0_net, 1, 28, 28, 16, 3, 3);

		bb::NeuralNetConvolutionPack<>		layer1_conv(&sub1_net, 16, 26, 26, 16, 3, 3);
		bb::NeuralNetMaxPooling<>			layer1_maxpol(16, 24, 24, 2, 2);

		bb::NeuralNetSparseBinaryAffine<>	layer2_affine(16 *12*12, 360 * 2);
		bb::NeuralNetSparseBinaryAffine<>	layer3_affine(360 * 2, 60 * 2);
		bb::NeuralNetSparseBinaryAffine<>	layer4_affine(60 * 2, 10 * 2);

		bb::NeuralNetBinaryToReal<float>	layer5_bin2real(10 * 2, 10);
		bb::NeuralNetSoftmax<>				layer5_softmax(10);

		bb::NeuralNet<> net;
		net.AddLayer(&layer0_real2bin);
		net.AddLayer(&layer0_conv);
		net.AddLayer(&layer1_conv);
		net.AddLayer(&layer1_maxpol);
		net.AddLayer(&layer2_affine);
		net.AddLayer(&layer3_affine);
		net.AddLayer(&layer4_affine);
		net.AddLayer(&layer5_bin2real);
		net.AddLayer(&layer5_softmax);

		for (int epoc = 0; epoc < epoc_size; ++epoc) {

			// 学習状況評価
			std::cout << get_time() << "s " << "epoc[" << epoc << "] accuracy : " << CalcAccuracy(net) << std::endl;

			for (size_t x_index = 0; x_index < m_train_images.size(); x_index += max_batch_size) {
				// 末尾のバッチサイズクリップ
				size_t batch_size = std::min(max_batch_size, m_train_images.size() - x_index);

				// データセット
				net.SetBatchSize(batch_size);
				for (size_t frame = 0; frame < batch_size; ++frame) {
					net.SetInputSignal(frame, m_train_images[x_index + frame]);
				}

				// 予測
				net.Forward();

				// 誤差逆伝播
				for (size_t frame = 0; frame < batch_size; ++frame) {
					auto signals = net.GetOutputSignal(frame);
					for (size_t node = 0; node < signals.size(); ++node) {
						signals[node] -= m_train_onehot[x_index + frame][node];
						signals[node] /= (float)batch_size;
					}
					net.SetOutputError(frame, signals);
				}
				net.Backward();

				// 更新
				net.Update();
			}
		}
		std::cout << "end\n" << std::endl;
	}

#if 0
	void RunLutSimpleConv(int epoc_size, size_t max_batch_size, int max_iteration = -1)
	{
		m_log << "start [LutSimpleConv]" << std::endl;
		reset_time();

		// 学習時と評価時で多重化数(乱数を変えて複数毎通して集計できるようにする)を変える
		int train_mux_size = 1;
		int test_mux_size = 3;

		// Conv用subネット構築 (3x3)
		bb::NeuralNetBinaryLut6<true>			sub0_lut0(1 * 3 * 3, 16);
		bb::NeuralNetGroup<>					sub0_net;
		sub0_net.AddLayer(&sub0_lut0);

		// Conv用subネット構築 (3x3)
		bb::NeuralNetBinaryLut6<true>			sub1_lut0(16 * 3 * 3, 64);
		bb::NeuralNetBinaryLut6<true>			sub1_lut1(64, 8);
		bb::NeuralNetGroup<>					sub1_net;
		sub1_net.AddLayer(&sub1_lut0);
		sub1_net.AddLayer(&sub1_lut1);

		// 実数版NET構築
		bb::NeuralNetRealToBinary<>				input_real2bin(1 * 28 * 28, 1 * 28 * 28);
		bb::NeuralNetConvolutionPack<bool>		layer0_conv(&sub0_net, 1, 28, 28, 16, 3, 3);
		bb::NeuralNetConvolutionPack<bool>		layer1_conv(&sub1_net, 16, 26, 26, 8, 3, 3);
		bb::NeuralNetMaxPooling<bool>			layer2_maxpol(8, 24, 24, 2, 2);
		bb::NeuralNetBinaryLut6<>				layer3_lut(8 * 12 * 12, 360);
		bb::NeuralNetBinaryLut6<>				layer4_lut(360, 60);
		bb::NeuralNetBinaryToReal<>				output_bin2rel(60, 10);
		auto last_lut_layer = &layer4_lut;

		bb::NeuralNet<>		net;
		net.AddLayer(&input_real2bin);
		net.AddLayer(&layer0_conv);
		net.AddLayer(&layer1_conv);
		net.AddLayer(&layer2_maxpol);
		net.AddLayer(&layer3_lut);
		net.AddLayer(&layer4_lut);
		//	net.AddLayer(&output_bin2rel);	// 学習時はunbinarize不要

		// 評価用NET構築(ノードは共有)
		bb::NeuralNet<>		eva_net;
		eva_net.AddLayer(&input_real2bin);
		eva_net.AddLayer(&layer0_conv);
		eva_net.AddLayer(&layer1_conv);
		eva_net.AddLayer(&layer2_maxpol);
		eva_net.AddLayer(&layer3_lut);
		eva_net.AddLayer(&layer4_lut);
		eva_net.AddLayer(&output_bin2rel);


		// 学習ループ
		int iteration = 0;
		for (int epoc = 0; epoc < epoc_size; ++epoc) {
			// 学習状況評価
			input_real2bin.InitializeCoeff(1);
			output_bin2rel.InitializeCoeff(1);
			//			eva_net.SetMuxSize(test_mux_size);
			auto accuracy = CalcAccuracy(eva_net);
			m_log << get_time() << "s " << "epoc[" << epoc << "] accuracy : " << accuracy << std::endl;

			for (size_t x_index = 0; x_index < m_train_images.size(); x_index += max_batch_size) {
				// 末尾のバッチサイズクリップ
				size_t batch_size = std::min(max_batch_size, m_train_images.size() - x_index);

				// バッチ学習データの作成
				std::vector< std::vector<float> >	batch_images(m_train_images.begin() + x_index, m_train_images.begin() + x_index + batch_size);
				std::vector< std::uint8_t >			batch_labels(m_train_labels.begin() + x_index, m_train_labels.begin() + x_index + batch_size);

				// データセット
				//				net.SetMuxSize(train_mux_size);
				net.SetBatchSize(batch_size);
				for (size_t frame = 0; frame < batch_size; ++frame) {
					net.SetInputSignal(frame, batch_images[frame]);
				}

				// 予測
				net.Forward();

				// バイナリ版フィードバック(力技学習)
				while (net.Feedback(last_lut_layer->GetOutputOnehotLoss<std::uint8_t, 10>(batch_labels)))
					;

				// 中間表示()
				input_real2bin.InitializeCoeff(1);
				output_bin2rel.InitializeCoeff(1);
				eva_net.SetMuxSize(test_mux_size);
				accuracy = CalcAccuracy(eva_net);
				m_log << get_time() << "s " << "epoc[" << epoc << "] accuracy : " << accuracy << std::endl;

				iteration++;
				if (max_iteration > 0 && iteration >= max_iteration) {
					goto loop_end;
				}
			}
		}
	loop_end:

		m_log << "end\n" << std::endl;
	}
#endif

#if 0
	void RunLutSimpleCnn(int epoc_size, size_t max_batch_size, int max_iteration = -1)
	{
		m_log << "start [LutSimpleCnn]" << std::endl;
		reset_time();

		// 学習時と評価時で多重化数(乱数を変えて複数毎通して集計できるようにする)を変える
		int train_mux_size = 1;
		int test_mux_size = 3;

		// Conv用subネット構築 (3x3)
		bb::NeuralNetBinaryLut6<true>		sub0_lut0(1 * 3 * 3, 16);
		bb::NeuralNetGroup<>				sub0_net;
		sub0_net.AddLayer(&sub0_lut0);

		// Conv用subネット構築 (3x3)
		bb::NeuralNetBinaryLut6<true>		sub1_lut0(16 * 3 * 3, 64);
		bb::NeuralNetBinaryLut6<true>		sub1_lut1(64, 16);
		bb::NeuralNetGroup<>				sub1_net;
		sub1_net.AddLayer(&sub1_lut0);
		sub1_net.AddLayer(&sub1_lut1);

		// Conv用subネット構築 (3x3)
		bb::NeuralNetBinaryLut6<true>		sub3_lut0(16 * 3 * 3, 64);
		bb::NeuralNetBinaryLut6<true>		sub3_lut1(64, 16);
		bb::NeuralNetGroup<>				sub3_net;
		sub3_net.AddLayer(&sub3_lut0);
		sub3_net.AddLayer(&sub3_lut1);

		// Conv用subネット構築 (3x3)
		bb::NeuralNetBinaryLut6<true>		sub4_lut0(16 * 3 * 3, 64);
		bb::NeuralNetBinaryLut6<true>		sub4_lut1(64, 16);
		bb::NeuralNetGroup<>				sub4_net;
		sub4_net.AddLayer(&sub4_lut0);
		sub4_net.AddLayer(&sub4_lut1);

		// Conv用subネット構築 (3x3)
		bb::NeuralNetBinaryLut6<true>		sub6_lut0(16 * 3 * 3, 64);
		bb::NeuralNetBinaryLut6<true>		sub6_lut1(64, 16);
		bb::NeuralNetGroup<>				sub6_net;
		sub6_net.AddLayer(&sub6_lut0);
		sub6_net.AddLayer(&sub6_lut1);


		// 実数版NET構築
		bb::NeuralNetRealToBinary<>				input_real2bin(1 * 28 * 28, 1 * 28 * 28);
		bb::NeuralNetConvolutionPack<bool>		layer0_conv(&sub0_net, 1, 28, 28, 16, 3, 3);
		bb::NeuralNetConvolutionPack<bool>		layer1_conv(&sub1_net, 16, 26, 26, 16, 3, 3);
		bb::NeuralNetMaxPooling<bool>			layer2_maxpol(16, 24, 24, 2, 2);
		bb::NeuralNetConvolutionPack<bool>		layer3_conv(&sub3_net, 16, 12, 12, 16, 3, 3);
		bb::NeuralNetConvolutionPack<bool>		layer4_conv(&sub4_net, 16, 10, 10, 16, 3, 3);
		bb::NeuralNetMaxPooling<bool>			layer5_maxpol(16, 8, 8, 2, 2);
		bb::NeuralNetConvolutionPack<bool>		layer6_conv(&sub6_net, 16, 4, 4, 16, 3, 3);
		bb::NeuralNetBinaryLut6<>				layer7_lut(16 * 2 * 2, 30);
		bb::NeuralNetBinaryToReal<>				output_bin2rel(30, 10);
		auto last_lut_layer = &layer7_lut;

		bb::NeuralNet<>		net;
		net.AddLayer(&input_real2bin);
		net.AddLayer(&layer0_conv);
		net.AddLayer(&layer1_conv);
		net.AddLayer(&layer2_maxpol);
		net.AddLayer(&layer3_conv);
		net.AddLayer(&layer4_conv);
		net.AddLayer(&layer5_maxpol);
		net.AddLayer(&layer6_conv);
		net.AddLayer(&layer7_lut);
		//	net.AddLayer(&output_bin2rel);	// 学習時はunbinarize不要

		// 評価用NET構築(ノードは共有)
		bb::NeuralNet<>		eva_net;
		eva_net.AddLayer(&input_real2bin);
		eva_net.AddLayer(&layer0_conv);
		eva_net.AddLayer(&layer1_conv);
		eva_net.AddLayer(&layer2_maxpol);
		eva_net.AddLayer(&layer3_conv);
		eva_net.AddLayer(&layer4_conv);
		eva_net.AddLayer(&layer5_maxpol);
		eva_net.AddLayer(&layer6_conv);
		eva_net.AddLayer(&layer7_lut);
		eva_net.AddLayer(&output_bin2rel);


		// 学習ループ
		int iteration = 0;
		for (int epoc = 0; epoc < epoc_size; ++epoc) {
			// 学習状況評価
			input_real2bin.InitializeCoeff(1);
			output_bin2rel.InitializeCoeff(1);
			eva_net.SetMuxSize(test_mux_size);
			auto accuracy = CalcAccuracy(eva_net);
			m_log << get_time() << "s " << "epoc[" << epoc << "] accuracy : " << accuracy << std::endl;

			for (size_t x_index = 0; x_index < m_train_images.size(); x_index += max_batch_size) {
				// 末尾のバッチサイズクリップ
				size_t batch_size = std::min(max_batch_size, m_train_images.size() - x_index);

				// バッチ学習データの作成
				std::vector< std::vector<float> >	batch_images(m_train_images.begin() + x_index, m_train_images.begin() + x_index + batch_size);
				std::vector< std::uint8_t >			batch_labels(m_train_labels.begin() + x_index, m_train_labels.begin() + x_index + batch_size);

				// データセット
				net.SetMuxSize(train_mux_size);
				net.SetBatchSize(batch_size);
				for (size_t frame = 0; frame < batch_size; ++frame) {
					net.SetInputSignal(frame, batch_images[frame]);
				}

				// 予測
				net.Forward();

				// バイナリ版フィードバック(力技学習)
				while (net.Feedback(last_lut_layer->GetOutputOnehotLoss<std::uint8_t, 10>(batch_labels)))
					;

				// 中間表示()
				if (iteration % 1 == 0) {
					input_real2bin.InitializeCoeff(1);
					output_bin2rel.InitializeCoeff(1);
					eva_net.SetMuxSize(test_mux_size);
					accuracy = CalcAccuracy(eva_net);
					m_log << get_time() << "s " << "epoc[" << epoc << "] accuracy : " << accuracy << std::endl;
				}

				iteration++;
				if (max_iteration > 0 && iteration >= max_iteration) {
					goto loop_end;
				}
			}
		}
	loop_end:

		m_log << "end\n" << std::endl;
	}
#endif

};




inline void testSetupLayerBuffer(bb::NeuralNetLayer<>& net)
{
	net.SetInputSignalBuffer(net.CreateInputSignalBuffer());
	net.SetInputErrorBuffer(net.CreateInputErrorBuffer());
	net.SetOutputSignalBuffer(net.CreateOutputSignalBuffer());
	net.SetOutputErrorBuffer(net.CreateOutputErrorBuffer());
}


#include "bb/NeuralNetConvExpandM.h"

static int tmp_d[64 * 1024 * 1024];
static int tmp_a[64 * 1024 * 1024];
static int tmp_b[64 * 1024 * 1024];



// メイン関数
int main()
{
	omp_set_num_threads(6);


#if 0
	// NeuralNetConvExpandを実践的なサイズで速度比較
	bb::NeuralNetConvExpand<> cnvexp(100, 28, 28, 3, 3);
	bb::NeuralNetConvExpandM<100, 28, 28, 3, 3> cnvexpM;

	cnvexp.SetBatchSize(256);
	cnvexpM.SetBatchSize(256);
	testSetupLayerBuffer(cnvexp);
	testSetupLayerBuffer(cnvexpM);

	std::chrono::system_clock::time_point  start, end;

	if (1) {
		// キャッシュを飛ばす
		for (int i = 0; i < sizeof(tmp_d) / sizeof(int); ++i) { tmp_d[i]++; }

		// 参考値
		start = std::chrono::system_clock::now();
		memcpy(tmp_a, tmp_b, sizeof(float) * 100 * 28 * 28 * 3 * 3);
		end = std::chrono::system_clock::now();
		double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << elapsed << std::endl;
	}

	if (1) {
		// キャッシュを飛ばす
		for (int i = 0; i < sizeof(tmp_d) / sizeof(int); ++i) { tmp_d[i]++; }

		start = std::chrono::system_clock::now();
		cnvexp.Forward();
		end = std::chrono::system_clock::now();
		double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "forward : " << elapsed << std::endl;

		start = std::chrono::system_clock::now();
		cnvexp.Forward();
		end = std::chrono::system_clock::now();
		elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "backward : " << elapsed << std::endl;
	}

	if (1) {
		// キャッシュを飛ばす
		for (int i = 0; i < sizeof(tmp_d) / sizeof(int); ++i) { tmp_d[i]++; }

		start = std::chrono::system_clock::now();
		cnvexpM.Forward();
		end = std::chrono::system_clock::now();
		double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << elapsed << std::endl;
	}
	//	getchar();
	return 0;
#endif






#ifdef _DEBUG
	std::cout << "!!!!DEBUG!!!!" << std::endl;
	int train_max_size = 128;
	int test_max_size = 128;
	int epoc_size = 16;
#else
	int train_max_size = -1;
	int test_max_size = -1;
	int epoc_size = 100;
#endif


	// 評価用クラスを作成
	EvaluateMnist	eva_mnist(train_max_size, test_max_size);

//	eva_mnist.RunSparseFullyCnn(1000, 128);

//	eva_mnist.RunFullyConvBinary(1000, 128);
//	eva_mnist.RunSimpleConvSigmoid(1000, 128);

//	eva_mnist.RunSimpleConvSigmoid(1000, 128, 0.01);
//	eva_mnist.RunSimpleConvPackSigmoid(1000, 128, 0.01);

//	eva_mnist.RunSimpleConvBinary(1000, 256, 0.1);
//	eva_mnist.RunFullyCnnBinary(1000, 256, 0.01);

#if 0
	// バイナリ6入力LUT版学習実験(重いです)
	eva_mnist.RunBinaryLut6WithBbruteForce(2, 8192, 8);
#endif

#if 0
	// 実数＆全接続(いわゆる古典的なニューラルネット)
	eva_mnist.RunDenseAffineSigmoid(16, 256, 1.0);
#endif

#if 0
	// 実数＆接続制限(接続だけLUT的にして中身のノードは実数)
	eva_mnist.RunFlatIlReal(16, 256);
#endif

#if 1
//	eva_mnist.RunSparseAffineSigmoid(5, 256);
	// 接続制限の実数で学習した後でバイナリにコピー
	eva_mnist.RunRealToBinary(8, 256);
#endif

	getchar();
	return 0;
}


