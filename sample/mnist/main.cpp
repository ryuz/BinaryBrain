#include <iostream>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <random>
#include <chrono>

#include <cereal/cereal.hpp>
#include <cereal/types/array.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/archives/json.hpp>

#include "bb/NeuralNet.h"
#include "bb/NeuralNetUtility.h"

#include "bb/NeuralNetRealToBinary.h"
#include "bb/NeuralNetBinaryToReal.h"
#include "bb/NeuralNetBinaryLut6.h"
#include "bb/NeuralNetBinaryLut6VerilogXilinx.h"

#include "bb/NeuralNetSigmoid.h"
#include "bb/NeuralNetReLU.h"
#include "bb/NeuralNetSoftmax.h"

#include "bb/NeuralNetAffine.h"
#include "bb/NeuralNetSparseAffine.h"
#include "bb/NeuralNetSparseAffineSigmoid.h"
#include "bb/NeuralNetDenseAffineSigmoid.h"

#include "bb/NeuralNetBatchNormalization.h"
#include "bb/NeuralNetBatchNormalizationAvx.h"

#include "bb/NeuralNetConvolution.h"
#include "bb/NeuralNetConvolutionPack.h"
#include "bb/NeuralNetMaxPooling.h"

#include "bb/NeuralNetOptimizerSgd.h"
#include "bb/NeuralNetOptimizerAdam.h"

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

	// ログ用出力
	std::ofstream		m_ofs_log;
	bb::ostream_tee		m_log;


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

		// 日時でログファイルオープン
		time_t time_now = time(NULL);
		struct tm tm;
		localtime_s(&tm, &time_now);
		std::stringstream ss;
		ss << "log_";
		ss << std::setfill('0') << std::setw(4) << tm.tm_year + 1900;
		ss << std::setfill('0') << std::setw(2) << tm.tm_mon + 1;
		ss << std::setfill('0') << std::setw(2) << tm.tm_mday;
		ss << std::setfill('0') << std::setw(2) << tm.tm_hour;
		ss << std::setfill('0') << std::setw(2) << tm.tm_min;
		ss << std::setfill('0') << std::setw(2) << tm.tm_sec;
		ss << ".txt";
		m_ofs_log.open(ss.str());
		if (m_ofs_log.is_open()) { 
			m_log.add(m_ofs_log);
		}
		m_log.add(std::cout);
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

	// 進捗表示
	void PrintProgress(float loss, size_t progress, size_t size)
	{
		size_t rate = progress * 100 / size;
		std::cout << "[" << rate << "% (" << progress << "/" << size << ")] loss : " << sqrt(loss) << "\r" << std::flush;
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
	// LUT6入力のバイナリ版の力技学習
	void RunBinaryLut6WithBbruteForce(int epoc_size, size_t max_batch_size, int max_iteration=-1)
	{
		m_log << "start [RunBinaryLut6WithBbruteForce]" << std::endl;
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
			m_log << get_time() << "s " << "epoc[" << epoc << "] accuracy : " << CalcAccuracy(net_eva) << std::endl;

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
				m_log << get_time() << "s " << "epoc[" << epoc << "] accuracy : " << CalcAccuracy(net_eva) << std::endl;

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

		m_log << "end\n" << std::endl;
	}


	// 実数(float)の全接続層で、フラットなネットを評価
	void RunSimpleDenseAffine(int epoc_size, size_t max_batch_size)
	{
		m_log << "start [SimpleDenseAffine]" << std::endl;
		reset_time();

		// 実数版NET構築
		bb::NeuralNetAffine<>  layer0_affine(28 * 28, 256);
		bb::NeuralNetSigmoid<> layer0_activation(256);
		bb::NeuralNetAffine<>  layer1_affine(256, 256);
		bb::NeuralNetSigmoid<> layer1_activation(256);
		bb::NeuralNetAffine<>  layer2_affine(256, 10);
		bb::NeuralNetSoftmax<> layer2_activation(10);

		bb::NeuralNet<> net;
		net.AddLayer(&layer0_affine);
		net.AddLayer(&layer0_activation);
		net.AddLayer(&layer1_affine);
		net.AddLayer(&layer1_activation);
		net.AddLayer(&layer2_affine);
		net.AddLayer(&layer2_activation);

		// オプティマイザ設定
		net.SetOptimizer(&bb::NeuralNetOptimizerAdam<>(0.001f, 0.9f, 0.999f));

		// 学習ループ
		for (int epoc = 0; epoc < epoc_size; ++epoc) {

			// 学習状況評価
			m_log << get_time() << "s " << "epoc[" << epoc << "] accuracy : " << CalcAccuracy(net) << std::endl;
			
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

			// Shuffle
			ShuffleTrainData();
		}
		m_log << "end\n" << std::endl;
	}

	// 実数(float)の全接続層で、フラットなネットを評価
	void RunSimpleDenseAffineBinary(int epoc_size, size_t max_batch_size)
	{
		m_log << "start [SimpleDenseAffineBinary]" << std::endl;
		reset_time();

		std::mt19937_64 mt(1);

		// 実数版NET構築
		size_t input_node_size = 1 * 28 * 28;
		size_t layer0_node_size = 10800;
		size_t layer1_node_size = 10800;
		size_t layer2_node_size = 3600;
		size_t layer3_node_size = 600;
		size_t layer4_node_size = 100;
		size_t output_node_size = 10;
		bb::NeuralNetBatchNormalization<>	input_batch_norm(input_node_size);
		bb::NeuralNetSigmoid<>				input_activation(input_node_size);

		bb::NeuralNetAffine<>				layer0_affine(input_node_size, layer0_node_size);
		bb::NeuralNetBatchNormalization<>	layer0_batch_norm(layer0_node_size);
		bb::NeuralNetSigmoid<>				layer0_activation(layer0_node_size);

		bb::NeuralNetAffine<>				layer1_affine(layer0_node_size, layer1_node_size);
		bb::NeuralNetBatchNormalization<>	layer1_batch_norm(layer1_node_size);
		bb::NeuralNetSigmoid<>				layer1_activation(layer1_node_size);

		bb::NeuralNetAffine<>				layer2_affine(layer1_node_size, layer2_node_size);
		bb::NeuralNetBatchNormalization<>	layer2_batch_norm(layer2_node_size);
		bb::NeuralNetSigmoid<>				layer2_activation(layer2_node_size);

		bb::NeuralNetAffine<>				layer3_affine(layer2_node_size, layer3_node_size);
		bb::NeuralNetBatchNormalization<>	layer3_batch_norm(layer3_node_size);
		bb::NeuralNetSigmoid<>				layer3_activation(layer3_node_size);

		bb::NeuralNetAffine<>				layer4_affine(layer3_node_size, layer4_node_size);
		bb::NeuralNetBatchNormalization<>	layer4_batch_norm(layer4_node_size);
		bb::NeuralNetSigmoid<>				layer4_activation(layer4_node_size);

		bb::NeuralNetBinaryToReal<float>	output_bin2real(layer4_node_size, output_node_size);
		bb::NeuralNetSoftmax<>				output_softmax(output_node_size);

		bb::NeuralNet<> net;
		net.AddLayer(&input_batch_norm);
		net.AddLayer(&input_activation);
		net.AddLayer(&layer0_affine);
		net.AddLayer(&layer0_batch_norm);
		net.AddLayer(&layer0_activation);
		net.AddLayer(&layer1_affine);
		net.AddLayer(&layer1_batch_norm);
		net.AddLayer(&layer1_activation);
		net.AddLayer(&layer2_affine);
		net.AddLayer(&layer2_batch_norm);
		net.AddLayer(&layer2_activation);
		net.AddLayer(&layer3_affine);
		net.AddLayer(&layer3_batch_norm);
		net.AddLayer(&layer3_activation);
		net.AddLayer(&layer4_affine);
		net.AddLayer(&layer4_batch_norm);
		net.AddLayer(&layer4_activation);
		net.AddLayer(&output_bin2real);
		net.AddLayer(&output_softmax);

		// オプティマイザ設定
		net.SetOptimizer(&bb::NeuralNetOptimizerAdam<>(0.001f, 0.9f, 0.999f));

		// バイナリ設定
		net.SetBinaryMode(true);

		// 学習ループ
		for (int epoc = 0; epoc < epoc_size; ++epoc) {

			// 学習状況評価
			m_log << get_time() << "s " << "epoc[" << epoc << "] accuracy : " << CalcAccuracy(net) << std::endl;

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
				float loss = 0;
				for (size_t frame = 0; frame < batch_size; ++frame) {
					auto signals = net.GetOutputSignal(frame);
					for (size_t node = 0; node < signals.size(); ++node) {
						signals[node] -= m_train_onehot[x_index + frame][node];
						signals[node] /= (float)batch_size;
						loss += signals[node] * signals[node];
					}
					net.SetOutputError(frame, signals);
				}
				net.Backward();

				// 更新
				net.Update();

				// 進捗表示
				PrintProgress(sqrt(loss), x_index + batch_size, m_train_images.size());
			}

			// Shuffle
			ShuffleTrainData();
		}
		m_log << "end\n" << std::endl;
	}
	

	// 実数(float)で6入力に制限したノードで層を形成してネットを評価
	void RunSimpleSparseAffine(int epoc_size, size_t max_batch_size, bool binary_mode)
	{
		m_log << "start [SimpleSparseAffine]" << std::endl;
		reset_time();

		// 実数版NET構築
		bb::NeuralNet<> net;
		size_t input_node_size = 28 * 28;
		size_t layer0_node_size = 10 * 6 * 6 * 6 * 3;
		size_t layer1_node_size = 10 * 6 * 6 * 6 * 3;
		size_t layer2_node_size = 10 * 6 * 6 * 6;
		size_t layer3_node_size = 10 * 6 * 6;
		size_t layer4_node_size = 10 * 6;
		size_t layer5_node_size = 10;
		size_t output_node_size = 10;
		bb::NeuralNetSparseAffine<>			layer0_affine(28 * 28, layer0_node_size);
		bb::NeuralNetBatchNormalization<>	layer0_norm(layer0_node_size);
		bb::NeuralNetSigmoid<>				layer0_sigmoid(layer0_node_size);
		bb::NeuralNetSparseAffine<>			layer1_affine(layer0_node_size, layer1_node_size);
		bb::NeuralNetBatchNormalization<>	layer1_norm(layer1_node_size);
		bb::NeuralNetSigmoid<>				layer1_sigmoid(layer1_node_size);
		bb::NeuralNetSparseAffine<>			layer2_affine(layer1_node_size, layer2_node_size);
		bb::NeuralNetSigmoid<>				layer2_sigmoid(layer2_node_size);
		bb::NeuralNetSparseAffine<>			layer3_affine(layer2_node_size, layer3_node_size);
		bb::NeuralNetSigmoid<>				layer3_sigmoid(layer3_node_size);
		bb::NeuralNetSparseAffine<>			layer4_affine(layer3_node_size, layer4_node_size);
		bb::NeuralNetSigmoid<>				layer4_sigmoid(layer4_node_size);
		bb::NeuralNetSparseAffine<>			layer5_affine(layer4_node_size, layer5_node_size);
		bb::NeuralNetSoftmax<>				layer5_softmax(layer5_node_size);
		net.AddLayer(&layer0_affine);
		net.AddLayer(&layer0_norm);
		net.AddLayer(&layer0_sigmoid);
		net.AddLayer(&layer1_affine);
		net.AddLayer(&layer1_norm);
		net.AddLayer(&layer1_sigmoid);
		net.AddLayer(&layer2_affine);
		net.AddLayer(&layer2_sigmoid);
		net.AddLayer(&layer3_affine);
		net.AddLayer(&layer3_sigmoid);
		net.AddLayer(&layer4_affine);
		net.AddLayer(&layer4_sigmoid);
		net.AddLayer(&layer5_affine);
		net.AddLayer(&layer5_softmax);

		// オプティマイザ設定
		net.SetOptimizer(&bb::NeuralNetOptimizerAdam<>(0.001f, 0.9f, 0.999f));

		// バイナリ設定
		m_log << "binary mode : " << binary_mode << std::endl;
		net.SetBinaryMode(binary_mode);

		// 学習ループ
		for (int epoc = 0; epoc < epoc_size; ++epoc) {

			// 学習状況評価
			auto accuracy = CalcAccuracy(net);
			m_log << get_time() << "s " << "epoc[" << epoc << "] accuracy : " << accuracy << std::endl;
			
			// ミニバッチ学習
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
				float loss = 0;
				for (size_t frame = 0; frame < batch_size; ++frame) {
					auto signals = net.GetOutputSignal(frame);
					for (size_t node = 0; node < signals.size(); ++node) {
						signals[node] -= m_train_onehot[x_index + frame][node];
						signals[node] /= (float)batch_size;
						loss += signals[node] * signals[node];
					}
					net.SetOutputError(frame, signals);
				}
				net.Backward();
				
				// 更新
				net.Update();

				// 進捗表示
				PrintProgress(sqrt(loss), x_index + batch_size, m_train_images.size());
			}

			// Shuffle
			ShuffleTrainData();
		}
		m_log << "end\n" << std::endl;
	}


	// 浮動小数点で学習させてバイナリにコピー
	void RunRealToBinary(int epoc_size, size_t max_batch_size)
	{
		m_log << "start [RealToBinary]" << std::endl;
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
		bb::NeuralNet<> real_net;
		bb::NeuralNetRealToBinary<float>	real_real2bin(input_node_size, input_node_size);
		bb::NeuralNetSparseAffineSigmoid<6>	real_affine0(input_node_size, layer0_node_size);
		bb::NeuralNetSparseAffineSigmoid<6>	real_affine1(layer0_node_size, layer1_node_size);
		bb::NeuralNetSparseAffineSigmoid<6>	real_affine2(layer1_node_size, layer2_node_size);
		bb::NeuralNetBinaryToReal<float>	real_bin2real(layer2_node_size, output_node_size);
		real_net.AddLayer(&real_real2bin);
		real_net.AddLayer(&real_affine0);
		real_net.AddLayer(&real_affine1);
		real_net.AddLayer(&real_affine2);
		real_net.AddLayer(&real_bin2real);

		// 活性化層のバイナリ化(sigmoid が binarizeとして動く)
		bool binary_mode = false;
		real_net.SetBinaryMode(binary_mode);

		// オプティマイザ設定
		real_net.SetOptimizer(&bb::NeuralNetOptimizerAdam<>(0.001f, 0.9f, 0.999f));

		// 実数で逆伝播で学習
		for (int epoc = 0; epoc < epoc_size; ++epoc) {

			// 学習状況評価
			real_net.SetMuxSize(test_mux_size);
			auto real_accuracy = CalcAccuracy(real_net);
			m_log << get_time() << "s " << "epoc[" << epoc << "] real_net accuracy : " << real_accuracy << std::endl;

			if ( !binary_mode && real_accuracy > 0.85) {
				binary_mode = true;
				real_net.SetBinaryMode(binary_mode);
				m_log << "(enable binary mode)" << std::endl;
				
				// 再評価
				real_accuracy = CalcAccuracy(real_net);
				m_log << get_time() << "s " << "epoc[" << epoc << "] real_net accuracy : " << real_accuracy << std::endl;
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

			// Shuffle
			ShuffleTrainData();
		}


		// バイナリ版NET構築
		bb::NeuralNet<>	bin_net;
		bb::NeuralNetRealToBinary<> bin_layer_real2bin(input_node_size, input_node_size);
		bb::NeuralNetBinaryLut6<>	bin_layer_lut0(input_node_size, layer0_node_size);
		bb::NeuralNetBinaryLut6<>	bin_layer_lut1(layer0_node_size, layer1_node_size);
		bb::NeuralNetBinaryLut6<>	bin_layer_lut2(layer1_node_size, layer2_node_size);
		bb::NeuralNetBinaryToReal<>	bin_layer_bin2real(layer2_node_size, output_node_size);
		bin_net.AddLayer(&bin_layer_real2bin);
		bin_net.AddLayer(&bin_layer_lut0);
		bin_net.AddLayer(&bin_layer_lut1);
		bin_net.AddLayer(&bin_layer_lut2);
		//		bin_net.AddLayer(&bin_layer_bin2real);

		// バイナリ評価用
		bb::NeuralNet<>	bin_net_eva;
		bin_net_eva.AddLayer(&bin_layer_real2bin);
		bin_net_eva.AddLayer(&bin_layer_lut0);
		bin_net_eva.AddLayer(&bin_layer_lut1);
		bin_net_eva.AddLayer(&bin_layer_lut2);
		bin_net_eva.AddLayer(&bin_layer_bin2real);

		// パラメータをコピー
		m_log << "[parameter copy] real-net -> binary-net" << std::endl;
		bin_layer_lut0.ImportLayer(real_affine0);
		bin_layer_lut1.ImportLayer(real_affine1);
		bin_layer_lut2.ImportLayer(real_affine2);

		// バイナリ版評価
		bin_net.SetMuxSize(test_mux_size);
	
		// 学習ループ
		max_batch_size = 8192;
		int max_iteration = 8;
		int iteration = 0;
		for (int epoc = 0; epoc < epoc_size; ++epoc) {
			// 学習状況評価
			bin_layer_real2bin.InitializeCoeff(1);
			bin_layer_bin2real.InitializeCoeff(1);
			bin_net_eva.SetMuxSize(test_mux_size);
			m_log << get_time() << "s " << "epoc[" << epoc << "] bin_net accuracy : " << CalcAccuracy(bin_net_eva) << std::endl;

			for (size_t x_index = 0; x_index < m_train_images.size(); x_index += max_batch_size) {
				// 末尾のバッチサイズクリップ
				size_t batch_size = std::min(max_batch_size, m_train_images.size() - x_index);

				// バッチ学習データの作成
				std::vector< std::vector<float> >	batch_images(m_train_images.begin() + x_index, m_train_images.begin() + x_index + batch_size);
				std::vector< std::uint8_t >			batch_labels(m_train_labels.begin() + x_index, m_train_labels.begin() + x_index + batch_size);

				// データセット
				bin_net.SetMuxSize(train_mux_size);
				bin_net.SetBatchSize(batch_size);
				for (size_t frame = 0; frame < batch_size; ++frame) {
					bin_net.SetInputSignal(frame, batch_images[frame]);
				}

				// 予測
				bin_net.Forward();

				// バイナリ版フィードバック(力技学習)
				while (bin_net.Feedback(bin_layer_lut2.GetOutputOnehotLoss<std::uint8_t, 10>(batch_labels)))
					;

				// 中間表示()
				bin_layer_real2bin.InitializeCoeff(1);
				bin_layer_bin2real.InitializeCoeff(1);
				bin_net_eva.SetMuxSize(test_mux_size);
				m_log << get_time() << "s " << "epoc[" << epoc << "] bin_net accuracy : " << CalcAccuracy(bin_net_eva) << std::endl;

				iteration++;
				if (max_iteration > 0 && iteration >= max_iteration) {
					goto loop_end;
				}
			}
		}
	loop_end:

		{
			std::ofstream ofs("lut_net.v");
			bb::NeuralNetBinaryLut6VerilogXilinx(ofs, bin_layer_lut0, "lutnet_layer0");
			bb::NeuralNetBinaryLut6VerilogXilinx(ofs, bin_layer_lut1, "lutnet_layer1");
			bb::NeuralNetBinaryLut6VerilogXilinx(ofs, bin_layer_lut2, "lutnet_layer2");
		}

		m_log << "end\n" << std::endl;
	}


	// 実数(float)の全接続層で、フラットなネットを評価
	void RunSimpleConvolution(int epoc_size, size_t max_batch_size, bool binary_mode)
	{
		m_log << "start [SimpleConvolution]" << std::endl;
		reset_time();

		std::mt19937_64 mt(1);

		// 実数版NET構築
		bb::NeuralNetBatchNormalization<>	input_batch_norm(1 * 28 * 28);
		bb::NeuralNetSigmoid<>				input_activation(1 * 28 * 28);

		bb::NeuralNetConvolution<>			layer0_convolution(1, 28, 28, 16, 3, 3);
		bb::NeuralNetBatchNormalization<>	layer0_batch_norm(16 * 26 * 26);
		bb::NeuralNetSigmoid<>				layer0_activation(16 * 26 * 26);

		bb::NeuralNetConvolution<>			layer1_convolution(16, 26, 26, 16, 3, 3);
		bb::NeuralNetBatchNormalization<>	layer1_batch_norm(16 * 24 * 24);
		bb::NeuralNetSigmoid<>				layer1_activation(16 * 24 * 24);

		bb::NeuralNetMaxPooling<>			layer2_pooling(16, 24, 24, 2, 2);

		bb::NeuralNetAffine<>				layer3_affine(16 * 12 * 12, 10);
		bb::NeuralNetBatchNormalization<>	layer3_batch_norm(10);
		bb::NeuralNetSigmoid<>				layer3_activation(10);

		bb::NeuralNetSoftmax<>				output_softmax(10);

		bb::NeuralNet<> net;
		net.AddLayer(&input_batch_norm);
		net.AddLayer(&input_activation);
		net.AddLayer(&layer0_convolution);
		net.AddLayer(&layer0_batch_norm);
		net.AddLayer(&layer0_activation);
		net.AddLayer(&layer1_convolution);
		net.AddLayer(&layer1_batch_norm);
		net.AddLayer(&layer1_activation);
		net.AddLayer(&layer2_pooling);
		net.AddLayer(&layer3_affine);
		net.AddLayer(&layer3_batch_norm);
		net.AddLayer(&layer3_activation);
		net.AddLayer(&output_softmax);

		// オプティマイザ設定
		net.SetOptimizer(&bb::NeuralNetOptimizerAdam<>(0.001f, 0.9f, 0.999f));

		// バイナリ設定
		m_log << "binary mode : " << binary_mode << std::endl;
		net.SetBinaryMode(binary_mode);

		// 学習ループ
		for (int epoc = 0; epoc < epoc_size; ++epoc) {

			// 学習状況評価
			m_log << get_time() << "s " << "epoc[" << epoc << "] accuracy : " << CalcAccuracy(net) << std::endl;

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
				float loss = 0;
				for (size_t frame = 0; frame < batch_size; ++frame) {
					auto signals = net.GetOutputSignal(frame);
					for (size_t node = 0; node < signals.size(); ++node) {
						signals[node] -= m_train_onehot[x_index + frame][node];
						signals[node] /= (float)batch_size;
						loss += signals[node] * signals[node];
					}
					net.SetOutputError(frame, signals);
				}
				net.Backward();

				// 進捗表示
				PrintProgress(loss, x_index + batch_size, m_train_images.size());

				// 更新
				net.Update();
			}

			// Shuffle
			bb::ShuffleDataSet(mt(), m_train_images, m_train_onehot);
		}
		m_log << "end\n" << std::endl;
	}

	void RunSparseFullyCnn(int epoc_size, size_t max_batch_size)
	{
		m_log << "start [SparseFullyCnn]" << std::endl;
		reset_time();

		// Conv用subネット構築 (3x3)
		bb::NeuralNetSparseAffineSigmoid<>	real_sub0_affine(1 * 3 * 3, 30);
		bb::NeuralNetGroup<>				real_sub0_net;
		real_sub0_net.AddLayer(&real_sub0_affine);

		// Conv用subネット構築 (3x3)
		bb::NeuralNetSparseAffineSigmoid<>	real_sub1_affine0(30 * 3 * 3, 180);
		bb::NeuralNetSparseAffineSigmoid<>	real_sub1_affine1(180, 30);
		bb::NeuralNetGroup<>				real_sub1_net;
		real_sub1_net.AddLayer(&real_sub1_affine0);
		real_sub1_net.AddLayer(&real_sub1_affine1);

		// Conv用subネット構築 (3x3)
		bb::NeuralNetSparseAffineSigmoid<>	real_sub3_affine0(30 * 3 * 3, 180);
		bb::NeuralNetSparseAffineSigmoid<>	real_sub3_affine1(180, 30);
		bb::NeuralNetGroup<>				real_sub3_net;
		real_sub3_net.AddLayer(&real_sub3_affine0);
		real_sub3_net.AddLayer(&real_sub3_affine1);

		// Conv用subネット構築 (3x3)
		bb::NeuralNetSparseAffineSigmoid<>	real_sub4_affine0(30 * 3 * 3, 180);
		bb::NeuralNetSparseAffineSigmoid<>	real_sub4_affine1(180, 30);
		bb::NeuralNetGroup<>				real_sub4_net;
		real_sub4_net.AddLayer(&real_sub4_affine0);
		real_sub4_net.AddLayer(&real_sub4_affine1);

		// Conv用subネット構築 (3x3)
		bb::NeuralNetSparseAffineSigmoid<>	real_sub6_affine0(30 * 3 * 3, 180);
		bb::NeuralNetSparseAffineSigmoid<>	real_sub6_affine1(180, 30);
		bb::NeuralNetGroup<>				real_sub6_net;
		real_sub6_net.AddLayer(&real_sub6_affine0);
		real_sub6_net.AddLayer(&real_sub6_affine1);

		// Conv用subネット構築 (3x3)
		bb::NeuralNetSparseAffineSigmoid<>	real_sub7_affine0(30 * 2 * 2, 180);
		bb::NeuralNetSparseAffineSigmoid<>	real_sub7_affine1(180, 30);
		bb::NeuralNetGroup<>				real_sub7_net;
		real_sub7_net.AddLayer(&real_sub7_affine0);
		real_sub7_net.AddLayer(&real_sub7_affine1);

		// 実数版NET構築
		bb::NeuralNetConvolutionPack<>		real_layer0_conv(&real_sub0_net, 1, 28, 28, 30, 3, 3);
		bb::NeuralNetConvolutionPack<>		real_layer1_conv(&real_sub1_net, 30, 26, 26, 30, 3, 3);
		bb::NeuralNetMaxPooling<>			real_layer2_maxpol(30, 24, 24, 2, 2);
		bb::NeuralNetConvolutionPack<>		real_layer3_conv(&real_sub3_net, 30, 12, 12, 30, 3, 3);
		bb::NeuralNetConvolutionPack<>		real_layer4_conv(&real_sub4_net, 30, 10, 10, 30, 3, 3);
		bb::NeuralNetMaxPooling<>			real_layer5_maxpol(30, 8, 8, 2, 2);
		bb::NeuralNetConvolutionPack<>		real_layer6_conv(&real_sub6_net, 30, 4, 4, 30, 3, 3);
		bb::NeuralNetConvolutionPack<>		real_layer7_conv(&real_sub7_net, 30, 2, 2, 30, 2, 2);
		bb::NeuralNetBinaryToReal<float>	real_layer8_bin2rel(30, 10);
		bb::NeuralNetSoftmax<>				real_layer9_softmax(10);

		bb::NeuralNet<>		real_net;
		real_net.AddLayer(&real_layer0_conv);
		real_net.AddLayer(&real_layer1_conv);
		real_net.AddLayer(&real_layer2_maxpol);
		real_net.AddLayer(&real_layer3_conv);
		real_net.AddLayer(&real_layer4_conv);
		real_net.AddLayer(&real_layer5_maxpol);
		real_net.AddLayer(&real_layer6_conv);
		real_net.AddLayer(&real_layer7_conv);
		real_net.AddLayer(&real_layer8_bin2rel);
		real_net.AddLayer(&real_layer9_softmax);

		real_net.SetOptimizer(&bb::NeuralNetOptimizerAdam<>(0.001f, 0.9f, 0.999f));
		real_net.SetBinaryMode(true);

		real_net.SetMuxSize(1);

		for (int epoc = 0; epoc < epoc_size; ++epoc) {

			// 学習状況評価
			m_log << get_time() << "s " << "epoc[" << epoc << "] accuracy : " << CalcAccuracy(real_net) << std::endl;

			size_t current_batch_size = 0;
			for (size_t x_index = 0; x_index < m_train_images.size(); x_index += max_batch_size) {
				// 末尾のバッチサイズクリップ
				size_t batch_size = std::min(max_batch_size, m_train_images.size() - x_index);
				if (current_batch_size != batch_size) {
					real_net.SetBatchSize(batch_size);
					current_batch_size = batch_size;
				}

				// データセット
				for (size_t frame = 0; frame < batch_size; ++frame) {
					real_net.SetInputSignal(frame, m_train_images[x_index + frame]);
				}

				// 予測
				real_net.Forward();

				// 誤差逆伝播
				float loss = 0;
				for (size_t frame = 0; frame < batch_size; ++frame) {
					auto signals = real_net.GetOutputSignal(frame);
					for (size_t node = 0; node < signals.size(); ++node) {
						signals[node] -= m_train_onehot[x_index + frame][node];
						signals[node] /= (float)batch_size;
						loss += signals[node] * signals[node];
					}
					real_net.SetOutputError(frame, signals);
				}
				real_net.Backward();

				// 更新
				real_net.Update();

				// 進捗表示
				PrintProgress(sqrt(loss), x_index + batch_size, m_train_images.size());
			}

			// Shuffle
			ShuffleTrainData();
		}
		m_log << "end\n" << std::endl;
	}


	void RunLutSimpleConv(int epoc_size, size_t max_batch_size, int max_iteration = -1)
	{
		m_log << "start [LutSimpleConv]" << std::endl;
		reset_time();

		// 学習時と評価時で多重化数(乱数を変えて複数毎通して集計できるようにする)を変える
		int train_mux_size = 1;
		int test_mux_size  = 3;

		// Conv用subネット構築 (3x3)
		bb::NeuralNetBinaryLut6<true>		sub0_lut0(1 * 3 * 3, 16);
		bb::NeuralNetGroup<>				sub0_net;
		sub0_net.AddLayer(&sub0_lut0);

		// Conv用subネット構築 (3x3)
		bb::NeuralNetBinaryLut6<true>		sub1_lut0(16 * 3 * 3, 64);
		bb::NeuralNetBinaryLut6<true>		sub1_lut1(64, 8);
		bb::NeuralNetGroup<>				sub1_net;
		sub1_net.AddLayer(&sub1_lut0);
		sub1_net.AddLayer(&sub1_lut1);

		// 実数版NET構築
		bb::NeuralNetRealToBinary<>				input_real2bin(1 * 28 * 28, 1 * 28 * 28);
		bb::NeuralNetConvolutionPack<bool>		layer0_conv(&sub0_net, 1, 28, 28, 16, 3, 3);
		bb::NeuralNetConvolutionPack<bool>		layer1_conv(&sub1_net, 16, 26, 26, 8, 3, 3);
		bb::NeuralNetMaxPooling<bool>			layer2_maxpol(8, 24, 24, 2, 2);
		bb::NeuralNetBinaryLut6<>				layer3_lut(8*12* 12, 360);
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
		bb::NeuralNetBinaryLut6<>				layer7_lut(16 *2*2, 30);
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
};



// メイン関数
int main()
{
	omp_set_num_threads(6);

#ifdef _DEBUG
	std::cout << "!!! Debug Version !!!" << std::endl;
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

	// 以下評価したいものを適当に切り替えてご使用ください

#if 0
	// バイナリ6入力LUT版学習実験(重いです)
	eva_mnist.RunBinaryLut6WithBbruteForce(2, 8192, 8);
#endif

#if 0
	// 実数＆全接続(いわゆる古典的なニューラルネット)
	eva_mnist.RunSimpleDenseAffine(16, 256);
#endif

#if 0
	eva_mnist.RunSimpleDenseAffineBinary(1600, 256);
#endif

#if 0
	// 実数＆接続制限(接続だけLUT的にして中身のノードは実数)
	eva_mnist.RunSimpleSparseAffine(1000, 256, true);
#endif

#if 0
	// 接続制限の実数で学習した後でバイナリにコピー
	eva_mnist.RunRealToBinary(16, 256);
#endif

#if 1
	eva_mnist.RunSimpleConvolution(1000, 256, false);
#endif

#if 0
	eva_mnist.RunSparseFullyCnn(1000, 256);
#endif

#if 0
	eva_mnist.RunLutSimpleConv(10000, 4096, 10000);
#endif

#if 0
	eva_mnist.RunLutSimpleCnn(10000, 1024, 10000);
#endif

	getchar();

	return 0;
}


