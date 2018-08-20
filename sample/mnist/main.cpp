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


float evaluation_net(bb::NeuralNet<>& net, std::vector< std::vector<float> >& images, std::vector<std::uint8_t>& labels);


void img_show(std::vector<float>& image)
{
	cv::Mat img(28, 28, CV_32F);
	memcpy(img.data, &image[0], sizeof(float) * 28 * 28);
	cv::imshow("img", img);
	cv::waitKey();
}

/*
std::vector<float> calc_onehot_loss(std::vector<std::uint8_t> label, bb::NeuralNetBuffer<> buf, size_t mux_size)
{
	int  frame_size = (int)buf.GetFrameSize();
	size_t node_size = buf.GetNodeSize();

	std::vector<float> vec_loss_x(frame_size);
	float* vec_loss = &vec_loss_x[0];

#pragma omp parallel for
	for (int frame = 0; frame < frame_size; ++frame) {
		vec_loss[frame] = 0;
		for (size_t node = 0; node < node_size; ++node) {
			if (label[frame / mux_size] == (node % 10)) {
				vec_loss[frame] += (buf.Get<bool>(frame, node) ? -1.00f : +1.00f);
			}
			else {
				vec_loss[frame] += (buf.Get<bool>(frame, node) ? +0.1f : -0.1f);
			}
		}
	}

	return vec_loss_x;
}
*/


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
	int loop_num = 1000;
#endif
	size_t batch_size = 1000;


	// MNISTデータ読み込み
	auto train_images = mnist_read_images_real<float>("train-images-idx3-ubyte", train_max_size);
	auto train_labels = mnist_read_labels("train-labels-idx1-ubyte", train_max_size);
	auto test_images = mnist_read_images_real<float>("t10k-images-idx3-ubyte", test_max_size);
	auto test_labels = mnist_read_labels("t10k-labels-idx1-ubyte", test_max_size);

//	auto train_onehot = mnist_read_labels_real<float, 10>("train-labels-idx1-ubyte", train_max_size);
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
	size_t bin_layer0_node_size = 360 * 8;
	size_t bin_layer1_node_size = 60 * 16;
	size_t bin_layer2_node_size = 10 * 16;
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
			std::cout << "real : " << evaluation_net(real_net, test_images, test_labels) << std::endl;
			std::cout << "bin  : " << evaluation_net(bin_net_eva, test_images, test_labels) << std::endl;
		}

		// test
		{
			std::ofstream ofs("test.v");
			bb::NeuralNetBinaryLut6VerilogXilinx<>(ofs, bin_lut0, "layer0_lut");
			bb::NeuralNetBinaryLut6VerilogXilinx<>(ofs, bin_lut1, "layer1_lut");
			bb::NeuralNetBinaryLut6VerilogXilinx<>(ofs, bin_lut2, "layer2_lut");
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

#if 1
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

#if 0
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


