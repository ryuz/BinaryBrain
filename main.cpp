#include <iostream>
#include <random>
#include <opencv2/opencv.hpp>
#include "NeuralNet.h"
#include "NeuralNetAffine.h"
#include "NeuralNetSigmoid.h"
#include "NeuralNetSoftmax.h"
#include "NeuralNetBinaryLut6.h"
#include "NeuralNetBinarize.h"
#include "NeuralNetUnbinarize.h"
#include "mnist_read.h"


float evaluation_net(bb::NeuralNet<>& net, std::vector< std::vector<float> >& images, std::vector<std::uint8_t>& labels);

void img_show(std::vector<float>& image)
{
	cv::Mat img(28, 28, CV_32F);
	memcpy(img.data, &image[0], sizeof(float) * 28 * 28);
	cv::imshow("img", img);
	cv::waitKey();
}

//std::unique_ptr< NeuralNetBufferAccessor<float, size_t> >	accessor;
//std::unique_ptr< NeuralNetBufferAccessor<T, INDEX> >	accessor;
//NeuralNetBufferAccessor<>*	accessor;


#include "NeuralNetType.h"



std::vector<float> calc_onehot_loss(std::vector<std::uint8_t> label, bb::NeuralNetBuffer<> buf, size_t mux_size)
{
	int  frame_size = buf.GetFrameSize();
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
	auto train_image = mnist_read_images_real<float>("train-images-idx3-ubyte", train_max_size);
	auto train_label = mnist_read_labels_real<float, 10>("train-labels-idx1-ubyte", train_max_size);
	auto train_label_u = mnist_read_labels("train-labels-idx1-ubyte", train_max_size);
	
	auto test_image = mnist_read_images_real<float>("t10k-images-idx3-ubyte", test_max_size);
	auto test_label = mnist_read_labels("t10k-labels-idx1-ubyte", test_max_size);


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
	std::vector<size_t> train_index(train_image.size());
	for (size_t i = 0; i < train_index.size(); ++i) {
		train_index[i] = i;
	}

	batch_size = std::min(batch_size, train_image.size());	
	for ( int loop = 0; loop < loop_num; ++loop) {
		// 学習状況評価
		if (loop % 1 == 0) {
//			std::cout << "real : " << evaluation_net(real_net, test_image, test_label) << std::endl;
			std::cout << "bin  : " << evaluation_net(bin_net_eva, test_image, test_label) << std::endl;
		}

		std::shuffle(train_index.begin(), train_index.end(), mt);

		real_net.SetBatchSize(batch_size);
		bin_net.SetBatchSize(batch_size);

		std::vector<std::uint8_t> train_label_batch;
		for (size_t frame = 0; frame < batch_size; ++frame) {
			real_net.SetInputValue(frame, train_image[train_index[frame]]);
			bin_net.SetInputValue(frame, train_image[train_index[frame]]);

			train_label_batch.push_back(train_label_u[train_index[frame]]);
		}

#if 0
		// 実数版誤差逆伝播
		real_net.Forward();
		for (size_t frame = 0; frame < batch_size; ++frame) {
			auto values = real_net.GetOutputValue(frame);

			for (size_t node = 0; node < values.size(); ++node) {
				values[node] -= train_label[train_index[frame]][node];
				values[node] /= (float)batch_size;
			}
			real_net.SetOutputError(frame, values);
		}
		real_net.Backward();
		real_net.Update(0.2);
#endif		

		// バイナリ版フィードバック
		bin_net.Forward();
		std::vector<float> vec_loss(batch_size*bin_mux_size);
		do {
			vec_loss = calc_onehot_loss(train_label_batch, bin_lut2.GetOutputValueBuffer(), bin_mux_size);

			/*
			auto buf = bin_lut2.GetOutputValueBuffer();
			size_t frame_size = batch_size*bin_mux_size;
			size_t node_size = buf.GetNodeSize();
			for (size_t frame = 0; frame < frame_size; ++frame) {
				vec_loss[frame] = 0;
				for (size_t node = 0; node < node_size; ++node) {
					if (train_label_u[train_index[frame / bin_mux_size]] == (node % 10)) {
						vec_loss[frame] += (buf.Get<bool>(frame, node) ? -1.00f : +1.00f);
					}
					else {
						vec_loss[frame] += (buf.Get<bool>(frame, node) ? +0.1f : -0.1f);
					}
				}
			}
			*/
#if 0
			for (size_t frame = 0; frame < batch_size*bin_mux_size; ++frame) {
				// Unbinarize前の段階で評価
				vec_loss[frame] = 0;
				auto buf = bin_lut2.GetOutputValueBuffer();
				for (size_t node = 0; node < buf.GetNodeSize(); ++node) {
					if (train_label[train_index[frame / bin_mux_size]][node % 10] > 0.5) {
						vec_loss[frame] += (buf.Get<bool>(frame, node) ? -1.00f : +1.00f);
					}
					else {
						vec_loss[frame] += (buf.Get<bool>(frame, node) ? +0.1f : -0.1f);
					}
				}
			}
#endif
		} while (bin_net.Feedback(vec_loss));
	}

	return 0;
}


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


