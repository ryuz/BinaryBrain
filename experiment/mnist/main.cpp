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
	
	// 時間計測
	std::chrono::system_clock::time_point m_base_time;
	void reset_time(void) {	m_base_time = std::chrono::system_clock::now(); }
	double get_time(void)
	{
		auto now_time = std::chrono::system_clock::now();
		return std::chrono::duration_cast<std::chrono::milliseconds>(now_time - m_base_time).count() / 1000.0;
	}

	// ネットの正解率評価
	double CalcAccuracy(bb::NeuralNet<>& net, std::vector< std::vector<float> >& images, std::vector<std::uint8_t>& labels)
	{
		// 評価サイズ設定
		net.SetBatchSize(images.size());

		// 評価画像設定
		for (size_t frame = 0; frame < images.size(); ++frame) {
			net.SetInputSignal(frame, images[frame]);
		}

		// 評価実施
		net.Forward(false);

		// 結果集計
		int ok_count = 0;
		for (size_t frame = 0; frame < images.size(); ++frame) {
			auto out_val = net.GetOutputSignal(frame);
			for (size_t i = 10; i < out_val.size(); i++) {
				out_val[i % 10] += out_val[i];
			}
			out_val.resize(10);
			int max_idx = bb::argmax<float>(out_val);
			ok_count += ((max_idx % 10) == (int)labels[frame] ? 1 : 0);
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
	}
	

	// LUT6入力のバイナリ版のフラットなネットを評価
	void RunFlatBinaryLut6(int epoc_size, size_t max_batch_size, int max_iteration=-1)
	{
		std::cout << "start [RunFlatBinaryLut6]" << std::endl;
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
		bb::NeuralNetRealToBinary<>	layer_real2bin(input_node_size, input_node_size, mt());
		bb::NeuralNetBinaryLut6<>	layer_lut0(input_node_size, layer0_node_size, mt());
		bb::NeuralNetBinaryLut6<>	layer_lut1(layer0_node_size, layer1_node_size, mt());
		bb::NeuralNetBinaryLut6<>	layer_lut2(layer1_node_size, layer2_node_size, mt());
		bb::NeuralNetBinaryToReal<>	layer_bin2real(layer2_node_size, output_node_size, mt());
		auto last_lut_layer = &layer_lut2;
		net.AddLayer(&layer_real2bin);
		net.AddLayer(&layer_lut0);
		net.AddLayer(&layer_lut1);
		net.AddLayer(&layer_lut2);
		//	net.AddLayer(&layer_bin2real);	// 学習時は bin2real 不要
		
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
				net.Update(learning_rate);
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
				net.Update(learning_rate);
			}
		}
		std::cout << "end\n" << std::endl;
	}


	// 実数(float)の全接続層で、フラットなネットを評価
	void RunFlatRealBatchNorm(int epoc_size, size_t max_batch_size, double learning_rate)
	{
		std::cout << "start [RunFlatRealBatchNorm]" << std::endl;
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
				net.Update(learning_rate);
			}
		}
		std::cout << "end\n" << std::endl;
	}


	void RunFlatRealBatchNormBinary(int epoc_size, size_t max_batch_size, double learning_rate)
	{
		std::cout << "start [RunFlatRealBatchNormBinary]" << std::endl;
		reset_time();

		// 実数版NET構築
		bb::NeuralNet<> net;
		bb::NeuralNetBinarize<>				in_binarize(28 * 28);
		bb::NeuralNetAffine<>				layer0_affine(28 * 28, 100);
		bb::NeuralNetBatchNormalization<>	layer0_batch_norm(100);
		bb::NeuralNetBinarize<>				layer0_binarize(100);
		bb::NeuralNetAffine<>				layer1_affine(100, 60);
		bb::NeuralNetBatchNormalization<>	layer1_batch_norm(60);
		bb::NeuralNetBinarize<>				layer1_binarize(60);
		net.AddLayer(&in_binarize);
		net.AddLayer(&layer0_affine);
		net.AddLayer(&layer0_batch_norm);
		net.AddLayer(&layer0_binarize);
		net.AddLayer(&layer1_affine);
		net.AddLayer(&layer1_batch_norm);
		net.AddLayer(&layer1_binarize);

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
						signals[node] -= m_train_onehot[x_index + frame][node % 10];
						signals[node] /= (float)batch_size;
					}
					net.SetOutputError(frame, signals);
				}
				net.Backward();

				// 更新
				net.Update(learning_rate);
			}
		}
		std::cout << "end\n" << std::endl;
	}


	// 実数(float)で6入力に制限ノードで層を形成して、フラットなネットを評価
	void RunFlatIlReal(int epoc_size, size_t max_batch_size)
	{
		std::cout << "start [RunFlatIlReal]" << std::endl;
		reset_time();

		// 実数版NET構築
		bb::NeuralNet<> net;
		size_t input_node_size = 28 * 28;
		size_t layer0_node_size = 10 * 6 * 6 * 3;
		size_t layer1_node_size = 10 * 6 * 6;
		size_t layer2_node_size = 10 * 6;
		size_t layer3_node_size = 10;
		size_t output_node_size = 10;
		bb::NeuralNetSparseAffine<>	layer0_affine(28 * 28, layer0_node_size);
		bb::NeuralNetSigmoid<>		layer0_sigmoid(layer0_node_size);
		bb::NeuralNetSparseAffine<> layer1_affine(layer0_node_size, layer1_node_size);
		bb::NeuralNetSigmoid<>		layer1_sigmoid(layer1_node_size);
		bb::NeuralNetSparseAffine<> layer2_affine(layer1_node_size, layer2_node_size);
		bb::NeuralNetSigmoid<>		layer2_sigmoid(layer2_node_size);
		bb::NeuralNetSparseAffine<> layer3_affine(layer2_node_size, layer3_node_size);
		bb::NeuralNetSoftmax<>		layer3_softmax(layer3_node_size);
		net.AddLayer(&layer0_affine);
		net.AddLayer(&layer0_sigmoid);
		net.AddLayer(&layer1_affine);
		net.AddLayer(&layer1_sigmoid);
		net.AddLayer(&layer2_affine);
		net.AddLayer(&layer2_sigmoid);
		net.AddLayer(&layer3_affine);
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
				net.Update(1.0);
			}
		}
		std::cout << "end\n" << std::endl;
	}


	// Binarized Neural Network
	void RunBnn(int epoc_size, size_t max_batch_size)
	{
		std::cout << "start [Binarized Neural Network]" << std::endl;
		reset_time();

		int train_mux_size = 1;
		int test_mux_size = 16;

		// 実数版NET構築
		bb::NeuralNet<> net;
		size_t input_node_size  = 28 * 28;
		size_t layer0_node_size = 360 * 2;
		size_t layer1_node_size = 60 * 4;
		size_t layer2_node_size = 10 * 8;
		size_t output_node_size = 10;
		bb::NeuralNetRealToBinary<float>		layer0_real2bin(28 * 28, 28 * 28);
		bb::NeuralNetSparseAffine<>	layer0_affine(28 * 28, layer0_node_size);
		bb::NeuralNetBatchNormalization<>		layer0_batch_norm(layer0_node_size);
		bb::NeuralNetBinarize<>					layer0_binarize(layer0_node_size);
		bb::NeuralNetSparseAffine<>	layer1_affine(layer0_node_size, layer1_node_size);
		bb::NeuralNetBatchNormalization<>		layer1_batch_norm(layer1_node_size);
		bb::NeuralNetBinarize<>					layer1_binarize(layer1_node_size);
		bb::NeuralNetSparseAffine<>	layer2_affine(layer1_node_size, layer2_node_size);
		bb::NeuralNetBatchNormalization<>		layer2_batch_norm(layer2_node_size);
		bb::NeuralNetBinarize<>					layer2_binarize(layer2_node_size);
		bb::NeuralNetBinaryToReal<float>		layer2_bin2real(layer2_node_size, output_node_size);
		net.AddLayer(&layer0_real2bin);
		net.AddLayer(&layer0_affine);
		net.AddLayer(&layer0_batch_norm);
		net.AddLayer(&layer0_binarize);
		net.AddLayer(&layer1_affine);
		net.AddLayer(&layer1_batch_norm);
		net.AddLayer(&layer1_binarize);
		net.AddLayer(&layer2_affine);
		net.AddLayer(&layer2_batch_norm);
		net.AddLayer(&layer2_binarize);
		net.AddLayer(&layer2_bin2real);

		for (int epoc = 0; epoc < epoc_size; ++epoc) {

			// 学習状況評価
			layer0_real2bin.InitializeCoeff(1);
			layer2_bin2real.InitializeCoeff(1);
			net.SetMuxSize(test_mux_size);
			std::cout << get_time() << "s " << "epoc[" << epoc << "] accuracy : " << CalcAccuracy(net) << std::endl;

			for (size_t x_index = 0; x_index < m_train_images.size(); x_index += max_batch_size) {
				// 末尾のバッチサイズクリップ
				size_t batch_size = std::min(max_batch_size, m_train_images.size() - x_index);

				// データセット
				net.SetMuxSize(train_mux_size);
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
				net.Update(1.0);
			}
		}
		std::cout << "end\n" << std::endl;
	}


	// Binarized Neural Network
	void RunBnnConnection6(int epoc_size, size_t max_batch_size)
	{
		std::cout << "start [RunBnnConnection6]" << std::endl;
		reset_time();

		// 実数版NET構築
		bb::NeuralNet<> net;
		size_t input_node_size = 28 * 28;
		size_t layer0_node_size = 10 * 6 * 6 * 3;
		size_t layer1_node_size = 10 * 6 * 6;
		size_t layer2_node_size = 10 * 6;
		size_t layer3_node_size = 10;
		size_t output_node_size = 10;
		bb::NeuralNetBatchNormalization<>	input_batch_norm(input_node_size);
		bb::NeuralNetBinarize<>				input_binarize(input_node_size);
		bb::NeuralNetSparseAffine<>			layer0_affine(input_node_size, layer0_node_size);
		bb::NeuralNetBatchNormalization<>	layer0_batch_norm(layer0_node_size);
		bb::NeuralNetBinarize<>				layer0_binarize(layer0_node_size);
		bb::NeuralNetSparseAffine<>			layer1_affine(layer0_node_size, layer1_node_size);
		bb::NeuralNetBatchNormalization<>	layer1_batch_norm(layer1_node_size);
		bb::NeuralNetBinarize<>				layer1_binarize(layer1_node_size);
		bb::NeuralNetSparseAffine<>			layer2_affine(layer1_node_size, layer2_node_size);
		bb::NeuralNetBatchNormalization<>	layer2_batch_norm(layer2_node_size);
		bb::NeuralNetBinarize<>				layer2_binarize(layer2_node_size);
		bb::NeuralNetSparseAffine<>			layer3_affine(layer2_node_size, layer3_node_size);
		bb::NeuralNetBatchNormalization<>	layer3_batch_norm(layer3_node_size);
		bb::NeuralNetSoftmax<>				layer3_softmax(layer3_node_size);
//		net.AddLayer(&input_batch_norm);
//		net.AddLayer(&input_binarize);
		net.AddLayer(&layer0_affine);
		net.AddLayer(&layer0_batch_norm);
		net.AddLayer(&layer0_binarize);
		net.AddLayer(&layer1_affine);
		net.AddLayer(&layer1_batch_norm);
		net.AddLayer(&layer1_binarize);
		net.AddLayer(&layer2_affine);
		net.AddLayer(&layer2_batch_norm);
		net.AddLayer(&layer2_binarize);
		net.AddLayer(&layer3_affine);
		net.AddLayer(&layer3_batch_norm);
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
						signals[node] -= m_train_onehot[x_index + frame][node % 10];
						signals[node] /= (float)batch_size;
					}
					net.SetOutputError(frame, signals);
				}
				net.Backward();

				// 更新
				net.Update(1.0);
			}
		}
		std::cout << "end\n" << std::endl;
	}


	// 浮動小数点で学習させてバイナリにコピー
	void RunFlatRealToBinary(int epoc_size, size_t max_batch_size)
	{
		std::cout << "start [RunFlatRealToBinary]" << std::endl;
		reset_time();

		std::mt19937_64 mt(1);

		int train_mux_size = 1;
		int test_mux_size = 16;


		// 層構成
		size_t input_node_size = 28 * 28;
		size_t layer0_node_size = 10 * 6 * 6 * 3;
		size_t layer1_node_size = 10 * 6 * 6;
		size_t layer2_node_size = 10 * 6;
		size_t layer3_node_size = 10;
		size_t output_node_size = 10;

		// 実数版NET構築
		bb::NeuralNet<> real_net;
		bb::NeuralNetSparseAffineBc<6>	real_layer0_affine(input_node_size, layer0_node_size);
		bb::NeuralNetSigmoid<>			real_layer0_sigmoid(layer0_node_size);
		bb::NeuralNetSparseAffineBc<6>  real_layer1_affine(layer0_node_size, layer1_node_size);
		bb::NeuralNetSigmoid<>			real_layer1_sigmoid(layer1_node_size);
		bb::NeuralNetSparseAffineBc<6>  real_layer2_affine(layer1_node_size, layer2_node_size);
		bb::NeuralNetSigmoid<>			real_layer2_sigmoid(layer2_node_size);
		bb::NeuralNetSparseAffineBc<6>  real_layer3_affine(layer2_node_size, layer3_node_size);
		bb::NeuralNetSoftmax<>			real_layer3_softmax(layer3_node_size);
		real_net.AddLayer(&real_layer0_affine);
		real_net.AddLayer(&real_layer0_sigmoid);
		real_net.AddLayer(&real_layer1_affine);
		real_net.AddLayer(&real_layer1_sigmoid);
		real_net.AddLayer(&real_layer2_affine);
		real_net.AddLayer(&real_layer2_sigmoid);
		real_net.AddLayer(&real_layer3_affine);
		real_net.AddLayer(&real_layer3_softmax);

		// バイナリ版NET構築
		bb::NeuralNet<>	bin_net;
		bb::NeuralNetRealToBinary<> bin_layer_real2bin(input_node_size, input_node_size);
		bb::NeuralNetBinaryLut6<>	bin_layer_lut0(input_node_size, layer0_node_size);
		bb::NeuralNetBinaryLut6<>	bin_layer_lut1(layer0_node_size, layer1_node_size);
		bb::NeuralNetBinaryLut6<>	bin_layer_lut2(layer1_node_size, layer2_node_size);
		bb::NeuralNetBinaryLut6<>	bin_layer_lut3(layer2_node_size, layer3_node_size);
		bb::NeuralNetBinaryToReal<> bin_layer_bin2real(layer3_node_size, output_node_size);
		bin_net.AddLayer(&bin_layer_real2bin);
		bin_net.AddLayer(&bin_layer_lut0);
		bin_net.AddLayer(&bin_layer_lut1);
		bin_net.AddLayer(&bin_layer_lut2);
		bin_net.AddLayer(&bin_layer_lut3);
		bin_net.AddLayer(&bin_layer_bin2real);

		for (int epoc = 0; epoc < epoc_size; ++epoc) {

			// 学習状況評価
			auto real_accuracy = CalcAccuracy(real_net);
			std::cout << get_time() << "s " << "epoc[" << epoc << "] accuracy : " << real_accuracy << std::endl;

			// ある程度学習したらバイナリを評価
			if (real_accuracy > 0.6) {
				// パラメータをコピー
				bin_layer_lut0.ImportLayer(real_layer0_affine);
				bin_layer_lut1.ImportLayer(real_layer1_affine);
				bin_layer_lut2.ImportLayer(real_layer2_affine);
				bin_layer_lut3.ImportLayer(real_layer3_affine);

				// バイナリ版評価
				bin_net.SetMuxSize(test_mux_size);
				std::cout << "epoc[" << epoc << "] bin_net accuracy : " << CalcAccuracy(bin_net) << std::endl;
			}

			for (size_t x_index = 0; x_index < m_train_images.size(); x_index += max_batch_size) {
				// 末尾のバッチサイズクリップ
				size_t batch_size = std::min(max_batch_size, m_train_images.size() - x_index);

				// 入力データ設定
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
				real_net.Update(1.0);
			}
		}
		std::cout << "end\n" << std::endl;
	}


	// 浮動小数点で学習させてバイナリにコピー2
	void RunFlatRealToBinary2(int epoc_size, size_t max_batch_size)
	{
		std::cout << "start [RunFlatRealToBinary2]" << std::endl;
		reset_time();

		std::mt19937_64 mt(1);

		int train_mux_size = 1;
		int test_mux_size = 16;
		
		// 層構成
		size_t input_node_size = 28 * 28;
		size_t layer0_node_size = 360 * 2;
		size_t layer1_node_size = 60 * 4;
		size_t layer2_node_size = 10 * 8;
		size_t output_node_size = 10;

		// 実数版NET構築
		bb::NeuralNet<> real_net;
		bb::NeuralNetRealToBinary<float>		real_layer0_real2bin(input_node_size, input_node_size);
		bb::NeuralNetSparseBinaryAffine<>		real_layer0_affine(input_node_size, layer0_node_size);
		bb::NeuralNetSparseBinaryAffine<>		real_layer1_affine(layer0_node_size, layer1_node_size);
		bb::NeuralNetSparseBinaryAffine<>		real_layer2_affine(layer1_node_size, layer2_node_size);
		bb::NeuralNetBinaryToReal<float>		real_layer2_bin2real(layer2_node_size, output_node_size);
		real_net.AddLayer(&real_layer0_real2bin);
		real_net.AddLayer(&real_layer0_affine);
		real_net.AddLayer(&real_layer1_affine);
		real_net.AddLayer(&real_layer2_affine);
		real_net.AddLayer(&real_layer2_bin2real);

		// バイナリ版NET構築
		bb::NeuralNet<>	bin_net;
		bb::NeuralNetRealToBinary<> bin_layer_real2bin(input_node_size, input_node_size);
		bb::NeuralNetBinaryLut6<>	bin_layer_lut0(input_node_size, layer0_node_size);
		bb::NeuralNetBinaryLut6<>	bin_layer_lut1(layer0_node_size, layer1_node_size);
		bb::NeuralNetBinaryLut6<>	bin_layer_lut2(layer1_node_size, layer2_node_size);
		bb::NeuralNetBinaryToReal<> bin_layer_bin2real(layer2_node_size, output_node_size);
		bin_net.AddLayer(&bin_layer_real2bin);
		bin_net.AddLayer(&bin_layer_lut0);
		bin_net.AddLayer(&bin_layer_lut1);
		bin_net.AddLayer(&bin_layer_lut2);
		bin_net.AddLayer(&bin_layer_bin2real);

		for (int epoc = 0; epoc < epoc_size; ++epoc) {

			// 学習状況評価
			real_net.SetMuxSize(test_mux_size);
			auto real_accuracy = CalcAccuracy(real_net);
			std::cout << get_time() << "s " << "epoc[" << epoc << "] real_net accuracy : " << real_accuracy << std::endl;

			// ある程度学習したらバイナリを評価
			if (real_accuracy > 0.7) {
				// パラメータをコピー
				bin_layer_lut0.ImportLayer(real_layer0_affine);
				bin_layer_lut1.ImportLayer(real_layer1_affine);
				bin_layer_lut2.ImportLayer(real_layer2_affine);

				// バイナリ版評価
				bin_net.SetMuxSize(test_mux_size);
				std::cout << get_time() << "s " << "epoc[" << epoc << "] bin_net accuracy : " << CalcAccuracy(bin_net) << std::endl;
			}

			for (size_t x_index = 0; x_index < m_train_images.size(); x_index += max_batch_size) {
				// 末尾のバッチサイズクリップ
				size_t batch_size = std::min(max_batch_size, m_train_images.size() - x_index);

				// 入力データ設定
				real_net.SetMuxSize(train_mux_size);
				real_net.SetBatchSize(batch_size);
				for (size_t frame = 0; frame < batch_size; ++frame) {
					real_net.SetInputSignal(frame, m_train_images[frame + x_index]);
				}

				// 予測
				real_net.Forward();

				// test
				if (0) {
					auto buf = real_layer1_affine.GetInputSignalBuffer();
					for (size_t frame = 0; frame < buf.GetFrameSize(); ++frame) {
						for (size_t node = 0; frame < buf.GetNodeSize(); ++node) {
							float v = buf.Get<float>(frame, node);
							std::cout << v << std::endl;
						}
					}
				}


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
				real_net.Update(1.0);
			}
		}
		std::cout << "end\n" << std::endl;
	}

	
	// 浮動小数点で学習させてバイナリにコピー3
	void RunFlatRealToBinary3(int epoc_size, size_t max_batch_size)
	{
		std::cout << "start [RunFlatRealToBinary3]" << std::endl;
		reset_time();

		std::mt19937_64 mt(1);

		int train_mux_size = 1;
		int test_mux_size = 16;

		// 層構成
		size_t input_node_size = 28 * 28;
		size_t layer0_node_size = 360 * 2;
		size_t layer1_node_size = 60 * 4;
		size_t layer2_node_size = 10 * 8;
		size_t output_node_size = 10;

		// 実数版NET構築
		bb::NeuralNet<> real_net;
		bb::NeuralNetRealToBinary<float>		real_layer0_real2bin(input_node_size, input_node_size);
		bb::NeuralNetSparseBinaryAffine<>		real_layer0_affine(input_node_size, layer0_node_size);
		bb::NeuralNetSparseBinaryAffine<>		real_layer1_affine(layer0_node_size, layer1_node_size);
		bb::NeuralNetSparseBinaryAffine<>		real_layer2_affine(layer1_node_size, layer2_node_size);

//		bb::NeuralNetLimitedConnectionAffine<>	real_layer0_affine(input_node_size, layer0_node_size);
//		bb::NeuralNetBatchNormalization<>		real_layer0_batch_norm(layer0_node_size);
//		bb::NeuralNetBinarize<>					real_layer0_binarize(layer0_node_size);
//		bb::NeuralNetLimitedConnectionAffine<>	real_layer1_affine(layer0_node_size, layer1_node_size);
//		bb::NeuralNetBatchNormalization<>		real_layer1_batch_norm(layer1_node_size);
//		bb::NeuralNetBinarize<>					real_layer1_binarize(layer1_node_size);
//		bb::NeuralNetLimitedConnectionAffine<>	real_layer2_affine(layer1_node_size, layer2_node_size);
//		bb::NeuralNetBatchNormalization<>		real_layer2_batch_norm(layer2_node_size);
//		bb::NeuralNetBinarize<>					real_layer2_binarize(layer2_node_size);
		bb::NeuralNetBinaryToReal<float>		real_layer2_bin2real(layer2_node_size, output_node_size);
		real_net.AddLayer(&real_layer0_real2bin);
		real_net.AddLayer(&real_layer0_affine);
//		real_net.AddLayer(&real_layer0_batch_norm);
//		real_net.AddLayer(&real_layer0_binarize);
		real_net.AddLayer(&real_layer1_affine);
//		real_net.AddLayer(&real_layer1_batch_norm);
//		real_net.AddLayer(&real_layer1_binarize);
		real_net.AddLayer(&real_layer2_affine);
//		real_net.AddLayer(&real_layer2_batch_norm);
//		real_net.AddLayer(&real_layer2_binarize);
		real_net.AddLayer(&real_layer2_bin2real);

		// バイナリ版NET構築
		bb::NeuralNet<>	bin_net;
		bb::NeuralNetRealToBinary<> bin_layer_real2bin(input_node_size, input_node_size);
		bb::NeuralNetBinaryLut6<>	bin_layer_lut0(input_node_size, layer0_node_size);
		bb::NeuralNetBinaryLut6<>	bin_layer_lut1(layer0_node_size, layer1_node_size);
		bb::NeuralNetBinaryLut6<>	bin_layer_lut2(layer1_node_size, layer2_node_size);
		bb::NeuralNetBinaryToReal<> bin_layer_bin2real(layer2_node_size, output_node_size);
		bin_net.AddLayer(&bin_layer_real2bin);
		bin_net.AddLayer(&bin_layer_lut0);
		bin_net.AddLayer(&bin_layer_lut1);
		bin_net.AddLayer(&bin_layer_lut2);
		bin_net.AddLayer(&bin_layer_bin2real);

		for (int epoc = 0; epoc < epoc_size; ++epoc) {
			// 学習状況評価
			real_net.InitializeCoeff(1);	// 評価時の乱数シードを固定する
			real_net.InitializeCoeff(1);
			real_net.SetMuxSize(test_mux_size);
			auto real_accuracy = CalcAccuracy(real_net);
			std::cout << get_time() << "s " << "epoc[" << epoc << "] real_net accuracy : " << real_accuracy << std::endl;

			// ある程度学習したらバイナリを評価
			if (real_accuracy > 0.7) {
				// パラメータをコピー
				bin_layer_lut0.ImportLayer(real_layer0_affine);
				bin_layer_lut1.ImportLayer(real_layer1_affine);
				bin_layer_lut2.ImportLayer(real_layer2_affine);

				// バイナリ版評価
				bin_layer_real2bin.InitializeCoeff(1);
				bin_layer_bin2real.InitializeCoeff(1);
				bin_net.SetMuxSize(test_mux_size);
				std::cout << get_time() << "s " << "epoc[" << epoc << "] bin_net accuracy : " << CalcAccuracy(bin_net) << std::endl;
			}

			for (size_t x_index = 0; x_index < m_train_images.size(); x_index += max_batch_size) {
				// 末尾のバッチサイズクリップ
				size_t batch_size = std::min(max_batch_size, m_train_images.size() - x_index);

				// 入力データ設定
				real_net.SetMuxSize(train_mux_size);
				real_net.SetBatchSize(batch_size);
				for (size_t frame = 0; frame < batch_size; ++frame) {
					real_net.SetInputSignal(frame, m_train_images[frame + x_index]);
				}

				// 予測
				real_net.Forward();

				// test
				if (0) {
					auto buf = real_layer1_affine.GetInputSignalBuffer();
					for (size_t frame = 0; frame < buf.GetFrameSize(); ++frame) {
						for (size_t node = 0; frame < buf.GetNodeSize(); ++node) {
							float v = buf.Get<float>(frame, node);
							std::cout << v << std::endl;
						}
					}
				}


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
				real_net.Update(1.0);
			}
		}
		std::cout << "end\n" << std::endl;
	}


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
				net.Update(learning_rate);
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
				net.Update(learning_rate);
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
				net.Update(learning_rate);
			}
		}
		std::cout << "end\n" << std::endl;
	}

	// 「ゼロから作る」の構成のSigmoid
	void RunSimpleConvSigmoid(int epoc_size, size_t max_batch_size, double learning_rate)
	{
		std::cout << "start [RunSimpleConvSigmoid]" << std::endl;
		reset_time();

		// 実数版NET構築
		bb::NeuralNet<> net;
		bb::NeuralNetConvolution<>  layer0_conv(1, 28, 28, 30, 5, 5);
		bb::NeuralNetSigmoid<>		layer0_sigmoid(24 * 24 * 30);
		bb::NeuralNetMaxPooling<>	layer0_maxpol(30, 24, 24, 2, 2);

		bb::NeuralNetAffine<>		layer1_affine(30 * 12 * 12, 100);
		bb::NeuralNetSigmoid<>		layer1_sigmoid(100);

		bb::NeuralNetAffine<>		layer2_affine(100, 10);
		bb::NeuralNetSoftmax<>		layer2_softmax(10);
		net.AddLayer(&layer0_conv);
		net.AddLayer(&layer0_sigmoid);
		net.AddLayer(&layer0_maxpol);
		net.AddLayer(&layer1_affine);
		net.AddLayer(&layer1_sigmoid);
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
				net.Update(learning_rate);
			}
		}
		std::cout << "end\n" << std::endl;
	}


	// Fully-CNN(動かない)
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
				net.Update(learning_rate);
			}
		}
		std::cout << "end\n" << std::endl;
	}
};


// メイン関数
int main()
{
	omp_set_num_threads(6);

#ifdef _DEBUG
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

//	eva_mnist.RunDenseAffineSigmoid(4, 256, 1.0);
//	eva_mnist.RunDenseAffineReLU(4, 256, 1.0);


	// 以下評価したいものを適当に切り替えてご使用ください
//	eva_mnist.RunBnnConnection6(64, 256*16);

//	eva_mnist.RunFlatRealBatchNorm(8, 256, 10.0);
//	eva_mnist.RunFlatRealBatchNormBinary(8, 256, 10.0);

//	eva_mnist.RunFlatReal(8, 256, 10.0);
//	getchar();

//	eva_mnist.RunConvolutionReal(8, 256, 10.0);
//	eva_mnist.RunCnnRealPre(100, 256, 2.0);
//	eva_mnist.RunConvSimple(100, 256, 10.0);

//	eva_mnist.RunBnn(16, 256);
//	eva_mnist.RunFlatRealToBinary2(100, 256);

//	eva_mnist.RunFlatRealToBinary3(100, 256);

//	eva_mnist.RunSimpleConv(16, 256, 0.1);
//	eva_mnist.RunFullyCnnReal(1000, 256, 0.01);

#if 1
	// バイナリ6入力LUT版学習実験(重いです)
	eva_mnist.RunFlatBinaryLut6(2, 8192, 8);
#endif

#if 0
	// 実数＆全接続(いわゆる古典的なニューラルネット)
	eva_mnist.RunFlatReal(16, 256, 1.0);
#endif

#if 0
	// 実数＆接続制限(接続だけLUT的にして中身のノードは実数)
	eva_mnist.RunFlatIlReal(16, 256);
#endif

#if 0
	// 接続制限の実数で学習した後でバイナリにコピー
	eva_mnist.RunFlatRealToBinary(100, 256);
#endif

	getchar();
	return 0;
}


