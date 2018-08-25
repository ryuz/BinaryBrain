#include <iostream>
#include <fstream>
#include <random>
#include <opencv2/opencv.hpp>
#include "bb/NeuralNet.h"
#include "bb/NeuralNetUtility.h"
#include "bb/NeuralNetAffine.h"
#include "bb/NeuralNetSigmoid.h"
#include "bb/NeuralNetSoftmax.h"
#include "bb/NeuralNetBinarize.h"
#include "bb/NeuralNetUnbinarize.h"
#include "bb/NeuralNetBinaryLut6.h"
#include "bb/NeuralNetBinaryLut6VerilogXilinx.h"
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

		// バイナリ版NET構築
		bb::NeuralNet<> net;
		size_t mux_size = 7;
		size_t input_node_size = 28 * 28;
		//	size_t layer0_node_size = 360 * 8;
		//	size_t layer1_node_size = 60 * 16;
		//	size_t layer2_node_size = 10 * 16;
		size_t layer0_node_size = 360 * 2;
		size_t layer1_node_size = 60 * 3;
		size_t layer2_node_size = 10 * 3;
		size_t output_node_size = 10;
		bb::NeuralNetBinarize<>   layer_binarize(input_node_size, input_node_size, mux_size);
		bb::NeuralNetBinaryLut6<> layer_lut0(input_node_size, layer0_node_size, mux_size);
		bb::NeuralNetBinaryLut6<> layer_lut1(layer0_node_size, layer1_node_size, mux_size);
		bb::NeuralNetBinaryLut6<> layer_lut2(layer1_node_size, layer2_node_size, mux_size);
		bb::NeuralNetUnbinarize<> layer_unbinarize(layer2_node_size, output_node_size, mux_size);
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

protected:
	// バッチ数分のサンプルをランダム選択
	void ShuffleTrainBatch(size_t batch_size, std::uint64_t seed)
	{
		std::mt19937_64 mt(seed);

		// シャッフル
		std::shuffle(m_train_batch_index.begin(), m_train_batch_index.end(), mt);

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
	size_t batch_size = 1000;

	EvaluateMnist	eva_mnist(train_max_size, test_max_size);

//	eva_mnist.RunFlatReal(loop_num, batch_size, 1);
	eva_mnist.RunFlatBinary(loop_num, batch_size, 1);

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

