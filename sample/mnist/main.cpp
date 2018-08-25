#include <iostream>
#include <fstream>
#include <random>

#include <opencv2/opencv.hpp>

#include <cereal/cereal.hpp>
#include <cereal/types/array.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/archives/json.hpp>

#include "bb/NeuralNet.h"
#include "bb/NeuralNetUtility.h"
#include "bb/NeuralNetAffine.h"
#include "bb/NeuralNetSigmoid.h"
#include "bb/NeuralNetSoftmax.h"
#include "bb/NeuralNetBinarize.h"
#include "bb/NeuralNetUnbinarize.h"
#include "bb/NeuralNetBinaryLut6.h"
#include "bb/NeuralNetBinaryLut6VerilogXilinx.h"
#include "bb/NeuralNetBinaryFilter.h"
#include "bb/ShuffleSet.h"
#include "mnist_read.h"


void img_show(std::vector<float>& image)
{
	cv::Mat img(28, 28, CV_32F);
	memcpy(img.data, &image[0], sizeof(float) * 28 * 28);
	cv::imshow("img", img);
	cv::waitKey();
}


// MNISTデータを使った評価用クラス
class EvaluateMnist
{
protected:
//	int train_max_size = 300;
//	int test_max_size = 10;
//	int test_rate = 1;
//	int loop_num = 2;

	// 評価用データセット
	std::vector< std::vector<float> >	m_test_images;
	std::vector< std::uint8_t >			m_test_labels;
	std::vector< std::vector<float> >	m_test_onehot;

	// 学習用データセット
	std::vector< std::vector<float> >	m_train_images;
	std::vector< std::uint8_t >			m_train_labels;
	std::vector< std::vector<float> >	m_train_onehot;

	// 学習用バッチ
	bb::ShuffleSet						m_shuffle_set;
	std::vector<size_t>					m_train_batch_index;
	std::vector< std::vector<float> >	m_train_batch_images;
	std::vector< std::uint8_t >			m_train_batch_labels;
	std::vector< std::vector<float> >	m_train_batch_onehot;
	
public:
	EvaluateMnist(int train_max_size = -1, int test_max_size = -1)
	{
		// MNISTデータ読み込み
		m_train_images = mnist_read_images_real<float>("train-images-idx3-ubyte", train_max_size);
		m_train_labels = mnist_read_labels("train-labels-idx1-ubyte", train_max_size);
		m_train_onehot = bb::LabelToOnehot<std::uint8_t, float>(m_train_labels, 10);

		m_test_images = mnist_read_images_real<float>("t10k-images-idx3-ubyte", test_max_size);
		m_test_labels = mnist_read_labels("t10k-labels-idx1-ubyte", test_max_size);
		m_test_onehot = bb::LabelToOnehot<std::uint8_t, float>(m_test_labels, 10);

		// インデックス作成
		m_shuffle_set.Setup(m_train_images.size());

		m_train_batch_index.resize(m_train_images.size());
		for (size_t i = 0; i < m_train_batch_index.size(); ++i) {
			m_train_batch_index[i] = i;
		}
	}

	// 浮動小数点版のフラットなネットを評価
	void RunFlatReal(int loop_num, size_t batch_size, int test_rate=1)
	{
		std::mt19937_64 mt(1);
		batch_size = std::min(batch_size, m_train_images.size());

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

		for (int loop = 0; loop < loop_num; ++loop) {
			// 学習状況評価
			if (loop % test_rate == 0) {
				std::cout << "test : " << TestNet(net) << std::endl;
			}

			// 学習データセットシャッフル
			ShuffleTrainBatch(batch_size, mt());

			// データセット
			net.SetBatchSize(batch_size);
			for (size_t frame = 0; frame < batch_size; ++frame) {
				net.SetInputValue(frame, m_train_batch_images[frame]);
			}

			// 予測
			net.Forward();

			// 誤差逆伝播
			for (size_t frame = 0; frame < batch_size; ++frame) {
				auto values = net.GetOutputValue(frame);
				for (size_t node = 0; node < values.size(); ++node) {
					values[node] -= m_train_batch_onehot[frame][node];
					values[node] /= (float)batch_size;
				}
				net.SetOutputError(frame, values);
			}
			net.Backward();
			
			// 更新
			net.Update(0.2);
		}
	}


	// バイナリ版のフラットなネットを評価
	void RunFlatBinary(int loop_num, size_t batch_size, int test_rate = 1)
	{
		std::mt19937_64 mt(1);
		batch_size = std::min(batch_size, m_train_images.size());

		int train_mux_size = 3;
		int test_mux_size  = 16;

		// バイナリ版NET構築
		bb::NeuralNet<> net;
		size_t input_node_size = 28 * 28;
		//	size_t layer0_node_size = 360 * 8;
		//	size_t layer1_node_size = 60 * 16;
		//	size_t layer2_node_size = 10 * 16;
		size_t layer0_node_size = 360 * 2;
		size_t layer1_node_size = 60 * 3;
		size_t layer2_node_size = 10 * 3;
		size_t output_node_size = 10;
		bb::NeuralNetBinarize<>   layer_binarize(input_node_size, input_node_size, mt());
		bb::NeuralNetBinaryLut6<> layer_lut0(input_node_size, layer0_node_size, mt());
		bb::NeuralNetBinaryLut6<> layer_lut1(layer0_node_size, layer1_node_size, mt());
		bb::NeuralNetBinaryLut6<> layer_lut2(layer1_node_size, layer2_node_size, mt());
		bb::NeuralNetUnbinarize<> layer_unbinarize(layer2_node_size, output_node_size, mt());
		net.AddLayer(&layer_binarize);
		net.AddLayer(&layer_lut0);
		net.AddLayer(&layer_lut1);
		net.AddLayer(&layer_lut2);
		//	net.AddLayer(&layer_unbinarize);

		// バイナリ版NET構築(評価用)
		bb::NeuralNet<> net_eva;
		net_eva.AddLayer(&layer_binarize);
		net_eva.AddLayer(&layer_lut0);
		net_eva.AddLayer(&layer_lut1);
		net_eva.AddLayer(&layer_lut2);
		net_eva.AddLayer(&layer_unbinarize);

		for (int loop = 0; loop < loop_num; ++loop) {
			// 学習状況評価
			if (loop % test_rate == 0) {
	//			layer_binarize.SetMuxSize(test_mux_size);
	//			layer_lut0.SetMuxSize(test_mux_size);
	//			layer_lut1.SetMuxSize(test_mux_size);
	//			layer_lut2.SetMuxSize(test_mux_size);
	//			layer_unbinarize.SetMuxSize(test_mux_size);
				net_eva.SetMuxSize(test_mux_size);
				std::cout << "test : " << TestNet(net_eva) << std::endl;
			}

			// test
			{
				std::ofstream ofs("test.v");
				bb::NeuralNetBinaryLut6VerilogXilinx(ofs, layer_lut0, "layer0_lut");
				bb::NeuralNetBinaryLut6VerilogXilinx(ofs, layer_lut1, "layer1_lut");
				bb::NeuralNetBinaryLut6VerilogXilinx(ofs, layer_lut2, "layer2_lut");
			}

			// 学習データセットシャッフル
			ShuffleTrainBatch(batch_size, mt());

			// データセット
	//		layer_binarize.SetMuxSize(train_mux_size);
	//		layer_lut0.SetMuxSize(train_mux_size);
	//		layer_lut1.SetMuxSize(train_mux_size);
	//		layer_lut2.SetMuxSize(train_mux_size);
	//		layer_unbinarize.SetMuxSize(train_mux_size);

			net_eva.SetMuxSize(train_mux_size);
			net.SetBatchSize(batch_size);
			for (size_t frame = 0; frame < batch_size; ++frame) {
				net.SetInputValue(frame, m_train_batch_images[frame]);
			}

			// 予測
			net.Forward();

			// バイナリ版フィードバック
			net.Forward();
			while (net.Feedback(layer_lut2.GetOutputOnehotLoss<std::uint8_t, 10>(m_train_batch_labels)))
				;
		}
	}


	// バイナリ版のCNNを評価
	void RunCnn1Binary(int loop_num, size_t batch_size, int test_rate = 1)
	{
		std::mt19937_64 mt(1);
		batch_size = std::min(batch_size, m_train_images.size());

		int train_mux_size = 3;
		int test_mux_size = 16;

		// layer0 畳み込み層
		size_t layer0_input_c_size = 1;
		size_t layer0_input_h_size = 28;
		size_t layer0_input_w_size = 28;
		size_t layer0_output_c_size = 2;
		size_t layer0_filter_h_size = 5;
		size_t layer0_filter_w_size = 5;
		size_t layer0_y_step = 1;
		size_t layer0_x_step = 1;

		size_t layer0_filter_input_node_size = layer0_input_c_size * layer0_filter_h_size * layer0_filter_h_size;
		size_t layer0_filter_lut0_node_size = 12;
		size_t layer0_filter_lut1_node_size = layer0_output_c_size;

		size_t input_node_size  = layer0_input_h_size * layer0_input_w_size;
		size_t layer0_node_size = layer0_output_c_size * (layer0_input_h_size - layer0_filter_h_size + 1) * (layer0_input_w_size - layer0_filter_w_size + 1);
		size_t layer1_node_size = 360 * 1;
		size_t layer2_node_size = 60 * 1;
		size_t layer3_node_size = 10 * 1;
		size_t output_node_size = 10;

		// 入力層(バイナライズ)
		bb::NeuralNetBinarize<>   layer_binarize(input_node_size, input_node_size, train_mux_size);

		// 畳み込み層0作成
		bb::NeuralNetBinaryLut6<true> layer0_filter_lut0(layer0_filter_input_node_size, layer0_filter_lut0_node_size, train_mux_size);
		bb::NeuralNetBinaryLut6<true> layer0_filter_lut1(layer0_filter_lut0_node_size, layer0_filter_lut1_node_size, train_mux_size);
		bb::NeuralNetGroup<> layer0_filter_net;
		layer0_filter_net.AddLayer(&layer0_filter_lut0);
		layer0_filter_net.AddLayer(&layer0_filter_lut1);
		bb::NeuralNetBinaryFilter<> layer0_cnv(&layer0_filter_net,
			layer0_input_c_size,
			layer0_input_h_size,
			layer0_input_w_size,
			layer0_output_c_size,
			layer0_filter_h_size,
			layer0_filter_w_size,
			layer0_y_step,
			layer0_x_step,
			train_mux_size);

		// LUT層
		bb::NeuralNetBinaryLut6<> layer1_lut(layer0_node_size, layer1_node_size, train_mux_size);
		bb::NeuralNetBinaryLut6<> layer2_lut(layer1_node_size, layer2_node_size, train_mux_size);
		bb::NeuralNetBinaryLut6<> layer3_lut(layer2_node_size, layer3_node_size, train_mux_size);
		
		// Unbinarize
		bb::NeuralNetUnbinarize<> layer_unbinarize(layer3_node_size, output_node_size, train_mux_size);
		
		// 学習用ネット
		bb::NeuralNet<> net;
		net.AddLayer(&layer_binarize);
		net.AddLayer(&layer0_cnv);
		net.AddLayer(&layer1_lut);
		net.AddLayer(&layer2_lut);
		net.AddLayer(&layer3_lut);
	//	net.AddLayer(&layer_unbinarize);

		// 評価用ネット
		bb::NeuralNet<> net_eva;
		net_eva.AddLayer(&layer_binarize);
		net_eva.AddLayer(&layer0_cnv);
		net_eva.AddLayer(&layer1_lut);
		net_eva.AddLayer(&layer2_lut);
		net_eva.AddLayer(&layer3_lut);
		net_eva.AddLayer(&layer_unbinarize);
		
		for (int loop = 0; loop < loop_num; ++loop) {
			// 学習状況評価
			if (loop % test_rate == 0) {
	//			layer_binarize.SetMuxSize(test_mux_size);
	//			layer_lut0.SetMuxSize(test_mux_size);
	//			layer_lut1.SetMuxSize(test_mux_size);
	//			layer_lut2.SetMuxSize(test_mux_size);
	//			layer_unbinarize.SetMuxSize(test_mux_size);

				std::cout << "test : " << TestNet(net_eva) << std::endl;
			}

			// 学習データセットシャッフル
			ShuffleTrainBatch(batch_size, mt());

			// データセット
	//		layer_binarize.SetMuxSize(train_mux_size);
	//		layer_lut0.SetMuxSize(train_mux_size);
	//		layer_lut1.SetMuxSize(train_mux_size);
	//		layer_lut2.SetMuxSize(train_mux_size);
	//		layer_unbinarize.SetMuxSize(train_mux_size);
			net.SetBatchSize(batch_size);
			for (size_t frame = 0; frame < batch_size; ++frame) {
				net.SetInputValue(frame, m_train_batch_images[frame]);
			}

			// 予測
			net.Forward();

			// バイナリ版フィードバック
			net.Forward();
			while (net.Feedback(layer3_lut.GetOutputOnehotLoss<std::uint8_t, 10>(m_train_batch_labels)))
				;
		}
	}



	// バイナリ版のCNNを評価
	void RunCnn2Binary(int loop_num, size_t batch_size, int test_rate = 1)
	{
		std::mt19937_64 mt(1);
		batch_size = std::min(batch_size, m_train_images.size());

		int train_mux_size = 3;
		int test_mux_size = 16;

		// layer0 畳み込み層相当
		size_t layer0_input_c_size = 1;
		size_t layer0_input_h_size = 28;
		size_t layer0_input_w_size = 28;
		size_t layer0_output_c_size = 8;
		size_t layer0_filter_h_size = 5;
		size_t layer0_filter_w_size = 5;
		size_t layer0_y_step = 1;
		size_t layer0_x_step = 1;
		size_t layer0_output_h_size = ((layer0_input_h_size - layer0_filter_h_size + 1) + (layer0_y_step - 1)) / layer0_y_step;
		size_t layer0_output_w_size = ((layer0_input_w_size - layer0_filter_w_size + 1) + (layer0_x_step - 1)) / layer0_x_step;

		size_t layer0_filter_input_node_size = layer0_input_c_size * layer0_filter_h_size * layer0_filter_w_size;
		size_t layer0_filter_lut0_node_size = 100;
		size_t layer0_filter_lut1_node_size = 48;
		size_t layer0_filter_lut2_node_size = layer0_output_c_size;


		// layer1 プーリング層相当
		size_t layer1_input_c_size = layer0_output_c_size;
		size_t layer1_input_h_size = layer0_output_h_size;
		size_t layer1_input_w_size = layer0_output_w_size;
		size_t layer1_output_c_size = 5;
		size_t layer1_filter_h_size = 2;
		size_t layer1_filter_w_size = 2;
		size_t layer1_y_step = 2;
		size_t layer1_x_step = 2;
		size_t layer1_output_h_size = ((layer1_input_h_size - layer1_filter_h_size + 1) + (layer1_y_step - 1)) / layer1_y_step;
		size_t layer1_output_w_size = ((layer1_input_w_size - layer1_filter_w_size + 1) + (layer1_x_step - 1)) / layer1_x_step;

		size_t layer1_filter_input_node_size = layer1_input_c_size * layer1_filter_h_size * layer1_filter_w_size;
		size_t layer1_filter_lut0_node_size = 30;
		size_t layer1_filter_lut1_node_size = layer1_output_c_size;
		
		// layer全体
		size_t input_node_size = layer0_input_h_size * layer0_input_w_size;
		size_t layer0_node_size = layer0_output_c_size * layer0_output_h_size * layer0_output_w_size;
		size_t layer1_node_size = layer1_output_c_size * layer1_output_h_size * layer1_output_w_size;
		size_t layer2_node_size = 360 * 2;
		size_t layer3_node_size = 60 * 3;
		size_t layer4_node_size = 10 * 3;
		size_t output_node_size = 10;

		// 入力層(バイナライズ)
		bb::NeuralNetBinarize<>   layer_binarize(input_node_size, input_node_size, train_mux_size);

		// layer0 畳み込み層作成
		bb::NeuralNetBinaryLut6<true> layer0_filter_lut0(layer0_filter_input_node_size, layer0_filter_lut0_node_size, train_mux_size);
		bb::NeuralNetBinaryLut6<true> layer0_filter_lut1(layer0_filter_lut0_node_size, layer0_filter_lut1_node_size, train_mux_size);
		bb::NeuralNetBinaryLut6<true> layer0_filter_lut2(layer0_filter_lut1_node_size, layer0_filter_lut2_node_size, train_mux_size);
		bb::NeuralNetGroup<> layer0_filter_net;
		layer0_filter_net.AddLayer(&layer0_filter_lut0);
		layer0_filter_net.AddLayer(&layer0_filter_lut1);
		layer0_filter_net.AddLayer(&layer0_filter_lut2);
		bb::NeuralNetBinaryFilter<> layer0_cnv(
			&layer0_filter_net,
			layer0_input_c_size,
			layer0_input_h_size,
			layer0_input_w_size,
			layer0_output_c_size,
			layer0_filter_h_size,
			layer0_filter_w_size,
			layer0_y_step,
			layer0_x_step,
			train_mux_size);

		// layer1 プーリング層作成
		bb::NeuralNetBinaryLut6<true> layer1_filter_lut0(layer1_filter_input_node_size, layer1_filter_lut0_node_size, train_mux_size);
		bb::NeuralNetBinaryLut6<true> layer1_filter_lut1(layer1_filter_lut0_node_size, layer1_filter_lut1_node_size, train_mux_size);
		bb::NeuralNetGroup<> layer1_filter_net;
		layer1_filter_net.AddLayer(&layer1_filter_lut0);
		layer1_filter_net.AddLayer(&layer1_filter_lut1);
		bb::NeuralNetBinaryFilter<> layer1_pol(
			&layer1_filter_net,
			layer1_input_c_size,
			layer1_input_h_size,
			layer1_input_w_size,
			layer1_output_c_size,
			layer1_filter_h_size,
			layer1_filter_w_size,
			layer1_y_step,
			layer1_x_step,
			train_mux_size);

		// Layer2-4 LUT層
		bb::NeuralNetBinaryLut6<> layer2_lut(layer1_node_size, layer2_node_size, train_mux_size);
		bb::NeuralNetBinaryLut6<> layer3_lut(layer2_node_size, layer3_node_size, train_mux_size);
		bb::NeuralNetBinaryLut6<> layer4_lut(layer3_node_size, layer4_node_size, train_mux_size);

		// バイナリの最終段
		bb::NeuralNetBinaryLut<>* lastBinLayer = &layer4_lut;

		// Unbinarize
		bb::NeuralNetUnbinarize<> layer_unbinarize(layer4_node_size, output_node_size, train_mux_size);

		// 学習用ネット
		bb::NeuralNet<> net;
		net.AddLayer(&layer_binarize);
		net.AddLayer(&layer0_cnv);
		net.AddLayer(&layer1_pol);
		net.AddLayer(&layer2_lut);
		net.AddLayer(&layer3_lut);
		net.AddLayer(&layer4_lut);
		//	net.AddLayer(&layer_unbinarize);

		// 評価用ネット
		bb::NeuralNet<> net_eva;
		net_eva.AddLayer(&layer_binarize);
		net_eva.AddLayer(&layer0_cnv);
		net_eva.AddLayer(&layer1_pol);
		net_eva.AddLayer(&layer2_lut);
		net_eva.AddLayer(&layer3_lut);
		net_eva.AddLayer(&layer4_lut);
		net_eva.AddLayer(&layer_unbinarize);

		for (int loop = 0; loop < loop_num; ++loop) {
			// 学習状況評価
			if (loop % test_rate == 0) {
				//			layer_binarize.SetMuxSize(test_mux_size);
				//			layer_lut0.SetMuxSize(test_mux_size);
				//			layer_lut1.SetMuxSize(test_mux_size);
				//			layer_lut2.SetMuxSize(test_mux_size);
				//			layer_unbinarize.SetMuxSize(test_mux_size);

				std::cout << "test : " << TestNet(net_eva) << std::endl;
			}

			// 学習データセットシャッフル
			ShuffleTrainBatch(batch_size, mt());

			// データセット
			//		layer_binarize.SetMuxSize(train_mux_size);
			//		layer_lut0.SetMuxSize(train_mux_size);
			//		layer_lut1.SetMuxSize(train_mux_size);
			//		layer_lut2.SetMuxSize(train_mux_size);
			//		layer_unbinarize.SetMuxSize(train_mux_size);
			net.SetBatchSize(batch_size);
			for (size_t frame = 0; frame < batch_size; ++frame) {
				net.SetInputValue(frame, m_train_batch_images[frame]);
			}

			// 予測
			net.Forward();

			// バイナリ版フィードバック
			net.Forward();
			while (net.Feedback(lastBinLayer->GetOutputOnehotLoss<std::uint8_t, 10>(m_train_batch_labels)))
				;
		}
	}


	// バイナリ版のCNNを評価
	void RunCnnBinary(int loop_num, size_t batch_size, int test_rate = 1)
	{
		std::mt19937_64 mt(1);
		batch_size = std::min(batch_size, m_train_images.size());

		int train_mux_size = 3;
		int test_mux_size = 16;

		// layer0 畳み込み層相当
		size_t layer0_input_c_size = 1;
		size_t layer0_input_h_size = 28;
		size_t layer0_input_w_size = 28;
		size_t layer0_output_c_size = 8;
		size_t layer0_filter_h_size = 5;
		size_t layer0_filter_w_size = 5;
		size_t layer0_y_step = 1;
		size_t layer0_x_step = 1;
		size_t layer0_output_h_size = ((layer0_input_h_size - layer0_filter_h_size + 1) + (layer0_y_step - 1)) / layer0_y_step;
		size_t layer0_output_w_size = ((layer0_input_w_size - layer0_filter_w_size + 1) + (layer0_x_step - 1)) / layer0_x_step;

		size_t layer0_filter_input_node_size = layer0_input_c_size * layer0_filter_h_size * layer0_filter_w_size;
		size_t layer0_filter_lut0_node_size = 100;
		size_t layer0_filter_lut1_node_size = 48;
		size_t layer0_filter_lut2_node_size = layer0_output_c_size;
		
		// layer1 プーリング層相当
		size_t layer1_input_c_size = layer0_output_c_size;
		size_t layer1_input_h_size = layer0_output_h_size;
		size_t layer1_input_w_size = layer0_output_w_size;
		size_t layer1_output_c_size = 5;
		size_t layer1_filter_h_size = 2;
		size_t layer1_filter_w_size = 2;
		size_t layer1_y_step = 2;
		size_t layer1_x_step = 2;
		size_t layer1_output_h_size = ((layer1_input_h_size - layer1_filter_h_size + 1) + (layer1_y_step - 1)) / layer1_y_step;
		size_t layer1_output_w_size = ((layer1_input_w_size - layer1_filter_w_size + 1) + (layer1_x_step - 1)) / layer1_x_step;

		size_t layer1_filter_input_node_size = layer1_input_c_size * layer1_filter_h_size * layer1_filter_w_size;
		size_t layer1_filter_lut0_node_size = 30;
		size_t layer1_filter_lut1_node_size = layer1_output_c_size;


		// layer2 畳み込み層相当
		size_t layer2_input_c_size = layer1_output_c_size;
		size_t layer2_input_h_size = layer1_output_h_size;
		size_t layer2_input_w_size = layer1_output_w_size;
		size_t layer2_output_c_size = 10;
		size_t layer2_filter_h_size = 5;
		size_t layer2_filter_w_size = 5;
		size_t layer2_y_step = 1;
		size_t layer2_x_step = 1;
		size_t layer2_output_h_size = ((layer2_input_h_size - layer2_filter_h_size + 1) + (layer2_y_step - 1)) / layer2_y_step;
		size_t layer2_output_w_size = ((layer2_input_w_size - layer2_filter_w_size + 1) + (layer2_x_step - 1)) / layer2_x_step;

		size_t layer2_filter_input_node_size = layer2_input_c_size * layer2_filter_h_size * layer2_filter_w_size;
		size_t layer2_filter_lut0_node_size = 100;
		size_t layer2_filter_lut1_node_size = 48;
		size_t layer2_filter_lut2_node_size = layer2_output_c_size;
		
		// layer3 プーリング層相当
		size_t layer3_input_c_size = layer2_output_c_size;
		size_t layer3_input_h_size = layer2_output_h_size;
		size_t layer3_input_w_size = layer2_output_w_size;
		size_t layer3_output_c_size = 5;
		size_t layer3_filter_h_size = 2;
		size_t layer3_filter_w_size = 2;
		size_t layer3_y_step = 2;
		size_t layer3_x_step = 2;
		size_t layer3_output_h_size = ((layer3_input_h_size - layer3_filter_h_size + 1) + (layer3_y_step - 1)) / layer3_y_step;
		size_t layer3_output_w_size = ((layer3_input_w_size - layer3_filter_w_size + 1) + (layer3_x_step - 1)) / layer3_x_step;
		
		size_t layer3_filter_input_node_size = layer3_input_c_size * layer3_filter_h_size * layer3_filter_w_size;
		size_t layer3_filter_lut0_node_size = 30;
		size_t layer3_filter_lut1_node_size = layer3_output_c_size;
		

		// layer全体
		size_t input_node_size = layer0_input_h_size * layer0_input_w_size;
		size_t layer0_node_size = layer0_output_c_size * layer0_output_h_size * layer0_output_w_size;
		size_t layer1_node_size = layer1_output_c_size * layer1_output_h_size * layer1_output_w_size;
		size_t layer2_node_size = layer2_output_c_size * layer2_output_h_size * layer2_output_w_size;
		size_t layer3_node_size = layer3_output_c_size * layer3_output_h_size * layer3_output_w_size;
		size_t layer4_node_size = 60 * 5;
		size_t layer5_node_size = 10 * 5;
		size_t output_node_size = 10;

		// 入力層(バイナライズ)
		bb::NeuralNetBinarize<>   layer_binarize(input_node_size, input_node_size, train_mux_size);

		// layer0 畳み込み層作成
		bb::NeuralNetBinaryLut6<true> layer0_filter_lut0(layer0_filter_input_node_size, layer0_filter_lut0_node_size, train_mux_size);
		bb::NeuralNetBinaryLut6<true> layer0_filter_lut1(layer0_filter_lut0_node_size, layer0_filter_lut1_node_size, train_mux_size);
		bb::NeuralNetBinaryLut6<true> layer0_filter_lut2(layer0_filter_lut1_node_size, layer0_filter_lut2_node_size, train_mux_size);
		bb::NeuralNetGroup<> layer0_filter_net;
		layer0_filter_net.AddLayer(&layer0_filter_lut0);
		layer0_filter_net.AddLayer(&layer0_filter_lut1);
		layer0_filter_net.AddLayer(&layer0_filter_lut2);
		bb::NeuralNetBinaryFilter<> layer0_cnv(
			&layer0_filter_net,
			layer0_input_c_size,
			layer0_input_h_size,
			layer0_input_w_size,
			layer0_output_c_size,
			layer0_filter_h_size,
			layer0_filter_w_size,
			layer0_y_step,
			layer0_x_step,
			train_mux_size);

		// layer1 プーリング層作成
		bb::NeuralNetBinaryLut6<true> layer1_filter_lut0(layer1_filter_input_node_size, layer1_filter_lut0_node_size, train_mux_size);
		bb::NeuralNetBinaryLut6<true> layer1_filter_lut1(layer1_filter_lut0_node_size, layer1_filter_lut1_node_size, train_mux_size);
		bb::NeuralNetGroup<> layer1_filter_net;
		layer1_filter_net.AddLayer(&layer1_filter_lut0);
		layer1_filter_net.AddLayer(&layer1_filter_lut1);
		bb::NeuralNetBinaryFilter<> layer1_pol(
			&layer1_filter_net,
			layer1_input_c_size,
			layer1_input_h_size,
			layer1_input_w_size,
			layer1_output_c_size,
			layer1_filter_h_size,
			layer1_filter_w_size,
			layer1_y_step,
			layer1_x_step,
			train_mux_size);

		// layer2 畳み込み層作成
		bb::NeuralNetBinaryLut6<true> layer2_filter_lut0(layer2_filter_input_node_size, layer2_filter_lut0_node_size, train_mux_size);
		bb::NeuralNetBinaryLut6<true> layer2_filter_lut1(layer2_filter_lut0_node_size, layer2_filter_lut1_node_size, train_mux_size);
		bb::NeuralNetBinaryLut6<true> layer2_filter_lut2(layer2_filter_lut1_node_size, layer2_filter_lut2_node_size, train_mux_size);
		bb::NeuralNetGroup<> layer2_filter_net;
		layer2_filter_net.AddLayer(&layer2_filter_lut0);
		layer2_filter_net.AddLayer(&layer2_filter_lut1);
		layer2_filter_net.AddLayer(&layer2_filter_lut2);
		bb::NeuralNetBinaryFilter<> layer2_cnv(
			&layer2_filter_net,
			layer2_input_c_size,
			layer2_input_h_size,
			layer2_input_w_size,
			layer2_output_c_size,
			layer2_filter_h_size,
			layer2_filter_w_size,
			layer2_y_step,
			layer2_x_step,
			train_mux_size);

		// layer3 プーリング層作成
		bb::NeuralNetBinaryLut6<true> layer3_filter_lut0(layer3_filter_input_node_size, layer3_filter_lut0_node_size, train_mux_size);
		bb::NeuralNetBinaryLut6<true> layer3_filter_lut1(layer3_filter_lut0_node_size, layer3_filter_lut1_node_size, train_mux_size);
		bb::NeuralNetGroup<> layer3_filter_net;
		layer3_filter_net.AddLayer(&layer3_filter_lut0);
		layer3_filter_net.AddLayer(&layer3_filter_lut1);
		bb::NeuralNetBinaryFilter<> layer3_pol(
			&layer3_filter_net,
			layer3_input_c_size,
			layer3_input_h_size,
			layer3_input_w_size,
			layer3_output_c_size,
			layer3_filter_h_size,
			layer3_filter_w_size,
			layer3_y_step,
			layer3_x_step,
			train_mux_size);

		// Layer4-5 LUT層
		bb::NeuralNetBinaryLut6<> layer4_lut(layer3_node_size, layer4_node_size, train_mux_size);
		bb::NeuralNetBinaryLut6<> layer5_lut(layer4_node_size, layer5_node_size, train_mux_size);

		// Unbinarize
		bb::NeuralNetUnbinarize<> layer_unbinarize(layer5_node_size, output_node_size, train_mux_size);

		// バイナリの最終段
		bb::NeuralNetBinaryLut<>* lastBinLayer = &layer5_lut;

		// 学習用ネット
		bb::NeuralNet<> net;
		net.AddLayer(&layer_binarize);
		net.AddLayer(&layer0_cnv);
		net.AddLayer(&layer1_pol);
		net.AddLayer(&layer2_cnv);
		net.AddLayer(&layer3_pol);
		net.AddLayer(&layer4_lut);
		net.AddLayer(&layer5_lut);
		//	net.AddLayer(&layer_unbinarize);

		// 評価用ネット
		bb::NeuralNet<> net_eva;
		net_eva.AddLayer(&layer_binarize);
		net_eva.AddLayer(&layer0_cnv);
		net_eva.AddLayer(&layer1_pol);
		net_eva.AddLayer(&layer2_cnv);
		net_eva.AddLayer(&layer3_pol);
		net_eva.AddLayer(&layer4_lut);
		net_eva.AddLayer(&layer5_lut);
		net_eva.AddLayer(&layer_unbinarize);

		for (int loop = 0; loop < loop_num; ++loop) {
			// 学習状況評価
			if (loop % test_rate == 0) {
				//			layer_binarize.SetMuxSize(test_mux_size);
				//			layer_lut0.SetMuxSize(test_mux_size);
				//			layer_lut1.SetMuxSize(test_mux_size);
				//			layer_lut2.SetMuxSize(test_mux_size);
				//			layer_unbinarize.SetMuxSize(test_mux_size);

				std::cout << "test : " << TestNet(net_eva) << std::endl;
			}

			// 学習データセットシャッフル
			ShuffleTrainBatch(batch_size, mt());

			// データセット
			//		layer_binarize.SetMuxSize(train_mux_size);
			//		layer_lut0.SetMuxSize(train_mux_size);
			//		layer_lut1.SetMuxSize(train_mux_size);
			//		layer_lut2.SetMuxSize(train_mux_size);
			//		layer_unbinarize.SetMuxSize(train_mux_size);
			net.SetBatchSize(batch_size);
			for (size_t frame = 0; frame < batch_size; ++frame) {
				net.SetInputValue(frame, m_train_batch_images[frame]);
			}

			// 予測
			net.Forward();

			// バイナリ版フィードバック
			net.Forward();
			while (net.Feedback(lastBinLayer->GetOutputOnehotLoss<std::uint8_t, 10>(m_train_batch_labels)))
				;
		}
	}



protected:
	// バッチ数分のサンプルをランダム選択
	void ShuffleTrainBatch(size_t batch_size, std::uint64_t seed)
	{
		std::mt19937_64 mt(seed);

		// シャッフル
	//	std::shuffle(m_train_batch_index.begin(), m_train_batch_index.end(), mt);
		m_train_batch_index = m_shuffle_set.GetRandomSet(batch_size);

		m_train_batch_images.resize(batch_size);
		m_train_batch_labels.resize(batch_size);
		m_train_batch_onehot.resize(batch_size);
		for (size_t frame = 0; frame < batch_size; ++frame) {
			m_train_batch_images[frame] = m_train_images[m_train_batch_index[frame]];
			m_train_batch_labels[frame] = m_train_labels[m_train_batch_index[frame]];
			m_train_batch_onehot[frame] = m_train_onehot[m_train_batch_index[frame]];
		}
	}


	// ネットの正解率テスト
	float TestNet(bb::NeuralNet<>& net, std::vector< std::vector<float> >& images, std::vector<std::uint8_t>& labels)
	{
		// 評価サイズ設定
		net.SetBatchSize(images.size());

		// 評価画像設定
		for (size_t frame = 0; frame < images.size(); ++frame) {
			net.SetInputValue(frame, images[frame]);
		}

		// 評価実施
		net.Forward();

		// 結果集計
		int ok_count = 0;
		for (size_t frame = 0; frame < images.size(); ++frame) {
			int max_idx = bb::argmax<float>(net.GetOutputValue(frame));
			ok_count += (max_idx == (int)labels[frame] ? 1 : 0);
		}

		//	std::cout << ok_count << " / " << images.size() << std::endl;

		return (float)ok_count / (float)images.size();
	}

	float TestNet(bb::NeuralNet<>& net)
	{
		return TestNet(net, m_test_images, m_test_labels);
	}
};



int main()
{
	omp_set_num_threads(6);

#ifdef _DEBUG
	int train_max_size = 300;
	int test_max_size = 10;
	int loop_num = 2;
#else
	int train_max_size = -1;
	int test_max_size = -1;
	int loop_num = 10000;
#endif
	size_t batch_size = 8192;

	EvaluateMnist	eva_mnist(train_max_size, test_max_size);

//	eva_mnist.RunFlatReal(loop_num, batch_size, 1);
	eva_mnist.RunFlatBinary(loop_num, batch_size, 1);
//	eva_mnist.RunCnn1Binary(loop_num, batch_size, 1);
//	eva_mnist.RunCnnBinary(loop_num, batch_size, 1);

	return 0;
}




#if 0

#define RUN_REAL	0
#define RUN_BINARY	1

int main()
{
	omp_set_num_threads(6);

	std::mt19937_64 mt(1);

#ifdef _DEBUG
	int train_max_size = 300;
	int test_max_size = 10;
	int loop_num = 2;
#else
	int train_max_size = -1;
	int test_max_size = -1;
	int loop_num = 10000;
#endif
	size_t batch_size = 1000;


	// MNISTデータ読み込み
	auto train_images = mnist_read_images_real<float>("train-images-idx3-ubyte", train_max_size);
	auto train_labels = mnist_read_labels("train-labels-idx1-ubyte", train_max_size);
	auto test_images = mnist_read_images_real<float>("t10k-images-idx3-ubyte", test_max_size);
	auto test_labels = mnist_read_labels("t10k-labels-idx1-ubyte", test_max_size);

	auto train_onehot = bb::LabelToOnehot<std::uint8_t, float>(train_labels, 10);
	

	// 実数版NET構築
	bb::NeuralNet<> real_net;
	bb::NeuralNetAffine<>  real_affine0(28*28, 100);
	bb::NeuralNetSigmoid<> real_sigmoid0(100);
	bb::NeuralNetAffine<>  real_affine1(100, 10);
	bb::NeuralNetSoftmax<> real_softmax1(10);
	real_net.AddLayer(&real_affine0);
	real_net.AddLayer(&real_sigmoid0);
	real_net.AddLayer(&real_affine1);
	real_net.AddLayer(&real_softmax1);

	// バイナリ版NET構築
	bb::NeuralNet<> bin_net;
	size_t bin_mux_size = 7;
	size_t bin_input_node_size  = 28*28;
//	size_t bin_layer0_node_size = 360 * 8;
//	size_t bin_layer1_node_size = 60 * 16;
//	size_t bin_layer2_node_size = 10 * 16;
	size_t bin_layer0_node_size = 360 * 2;
	size_t bin_layer1_node_size = 60 * 3;
	size_t bin_layer2_node_size = 10 * 3;
	size_t bin_output_node_size = 10;
	bb::NeuralNetBinarize<>   bin_binarize(bin_input_node_size, bin_input_node_size, bin_mux_size);
	bb::NeuralNetBinaryLut6<> bin_lut0(bin_input_node_size, bin_layer0_node_size, bin_mux_size);
	bb::NeuralNetBinaryLut6<> bin_lut1(bin_layer0_node_size, bin_layer1_node_size, bin_mux_size);
	bb::NeuralNetBinaryLut6<> bin_lut2(bin_layer1_node_size, bin_layer2_node_size, bin_mux_size);
	bb::NeuralNetUnbinarize<> bin_unbinarize(bin_layer2_node_size, bin_output_node_size, bin_mux_size);
	bin_net.AddLayer(&bin_binarize);
	bin_net.AddLayer(&bin_lut0);
	bin_net.AddLayer(&bin_lut1);
	bin_net.AddLayer(&bin_lut2);
//	bin_net.AddLayer(&bin_unbinarize);

	// バイナリ版NET構築(評価用)
	bb::NeuralNet<> bin_net_eva;
	bin_net_eva.AddLayer(&bin_binarize);
	bin_net_eva.AddLayer(&bin_lut0);
	bin_net_eva.AddLayer(&bin_lut1);
	bin_net_eva.AddLayer(&bin_lut2);
	bin_net_eva.AddLayer(&bin_unbinarize);

	// インデックス作成
	std::vector<size_t> train_index(train_images.size());
	for (size_t i = 0; i < train_index.size(); ++i) {
		train_index[i] = i;
	}

	batch_size = std::min(batch_size, train_images.size());
	for ( int loop = 0; loop < loop_num; ++loop) {
		// 学習状況評価
		if (loop % 1 == 0) {
#if RUN_REAL
			std::cout << "real : " << evaluation_net(real_net, test_images, test_labels) << std::endl;
#endif		
#if RUN_BINARY
			std::cout << "bin  : " << evaluation_net(bin_net_eva, test_images, test_labels) << std::endl;
#endif
		}

		// test
		{
			std::ofstream ofs("test.v");
			bb::NeuralNetBinaryLut6VerilogXilinx(ofs, bin_lut0, "layer0_lut");
			bb::NeuralNetBinaryLut6VerilogXilinx(ofs, bin_lut1, "layer1_lut");
			bb::NeuralNetBinaryLut6VerilogXilinx(ofs, bin_lut2, "layer2_lut");
		}

		std::shuffle(train_index.begin(), train_index.end(), mt);

		real_net.SetBatchSize(batch_size);
		bin_net.SetBatchSize(batch_size);

		std::vector<std::uint8_t> train_label_batch;
		for (size_t frame = 0; frame < batch_size; ++frame) {
			real_net.SetInputValue(frame, train_images[train_index[frame]]);
			bin_net.SetInputValue(frame, train_images[train_index[frame]]);

			train_label_batch.push_back(train_labels[train_index[frame]]);
		}

#if RUN_REAL
		// 実数版誤差逆伝播
		real_net.Forward();
		for (size_t frame = 0; frame < batch_size; ++frame) {
			auto values = real_net.GetOutputValue(frame);

			for (size_t node = 0; node < values.size(); ++node) {
				values[node] -= train_onehot[train_index[frame]][node];
				values[node] /= (float)batch_size;
			}
			real_net.SetOutputError(frame, values);
		}
		real_net.Backward();
		real_net.Update(0.2);
#endif		

#if RUN_BINARY
		// バイナリ版フィードバック
		bin_net.Forward();
		while (bin_net.Feedback(bin_lut2.GetOutputOnehotLoss<std::uint8_t, 10>(train_label_batch)))
			;
#endif
	}

	return 0;
}


// テスト用の画像で正解率を評価
float evaluation_net(bb::NeuralNet<>& net, std::vector< std::vector<float> >& images, std::vector<std::uint8_t>& labels)
{
	// 評価サイズ設定
	net.SetBatchSize(images.size());
	
	// 評価画像設定
	for ( size_t frame = 0; frame < images.size(); ++frame ){
		net.SetInputValue(frame, images[frame]);
	}

	// 評価実施
	net.Forward();

	// 結果集計
	int ok_count = 0;
	for (size_t frame = 0; frame < images.size(); ++frame) {
		int max_idx = bb::argmax<float>(net.GetOutputValue(frame));
		ok_count += (max_idx == (int)labels[frame] ? 1 : 0);
	}

//	std::cout << ok_count << " / " << images.size() << std::endl;

	return (float)ok_count / (float)images.size();
}


#endif

