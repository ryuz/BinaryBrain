#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>

#include <opencv2/opencv.hpp>


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



std::vector<std::uint8_t>				train_labels;
std::vector< std::vector<float> >		train_images;
std::vector< std::vector<float> >		train_onehot;

std::vector<std::uint8_t>				test_labels;
std::vector< std::vector<float> >		test_images;
std::vector< std::vector<float> >		test_onehot;


void read_cifer10(std::istream& is, std::vector<std::uint8_t>& vec_label, std::vector< std::vector<std::uint8_t> >& vec_image)
{
	while (!is.eof()) {
		std::uint8_t label;
		is.read((char*)&label, 1);
		if (is.eof()) { break; }

		vec_label.push_back(label);

		std::vector<std::uint8_t> image(32 * 32 * 3);
		is.read((char*)&image[0], 32*32*3);
		vec_image.push_back(image);
	}
}

void read_cifer10(std::string filename, std::vector<std::uint8_t>& vec_label, std::vector< std::vector<std::uint8_t> >& vec_image)
{
	std::ifstream ifs(filename, std::ios::binary);
	read_cifer10(ifs, vec_label, vec_image);
}


// 時間計測
std::chrono::system_clock::time_point m_base_time;

void reset_time(void) {
	m_base_time = std::chrono::system_clock::now();
}

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
	return CalcAccuracy(net, test_images, test_labels);
}


// 実数(float)の全接続層で、フラットなネットを評価
void RunDenseAffineSigmoid(int epoc_size, size_t max_batch_size, double learning_rate)
{
	std::cout << "start [RunDenseAffineSigmoid]" << std::endl;
	reset_time();

	// 実数版NET構築
	bb::NeuralNet<> net;
	bb::NeuralNetAffine<>  layer0_affine(3 * 32 * 32, 200);
	bb::NeuralNetSigmoid<> layer0_sigmoid(200);
	bb::NeuralNetAffine<>  layer1_affine(200, 10);
	bb::NeuralNetSoftmax<> layer1_softmax(10);
	net.AddLayer(&layer0_affine);
	net.AddLayer(&layer0_sigmoid);
	net.AddLayer(&layer1_affine);
	net.AddLayer(&layer1_softmax);

	for (int epoc = 0; epoc < epoc_size; ++epoc) {

		// 学習状況評価
		std::cout << get_time() << "s " << "epoc[" << epoc << "] accuracy : " << CalcAccuracy(net) << std::endl;

		for (size_t x_index = 0; x_index < train_images.size(); x_index += max_batch_size) {
			// 末尾のバッチサイズクリップ
			size_t batch_size = std::min(max_batch_size, train_images.size() - x_index);

			// データセット
			net.SetBatchSize(batch_size);
			for (size_t frame = 0; frame < batch_size; ++frame) {
				net.SetInputSignal(frame, train_images[x_index + frame]);
			}

			// 予測
			net.Forward();

			// 誤差逆伝播
			for (size_t frame = 0; frame < batch_size; ++frame) {
				auto signals = net.GetOutputSignal(frame);
				for (size_t node = 0; node < signals.size(); ++node) {
					signals[node] -= train_onehot[x_index + frame][node];
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


// LUT6入力のバイナリ版のフラットなネットを評価
void RunFlatBinaryLut6(int epoc_size, size_t max_batch_size, int max_iteration = -1)
{
	std::cout << "start [RunFlatBinaryLut6]" << std::endl;
	reset_time();

	std::mt19937_64 mt(1);

	// 学習時と評価時で多重化数(乱数を変えて複数毎通して集計できるようにする)を変える
	int train_mux_size = 1;
	int test_mux_size = 16;

	// 層構成定義
	size_t input_node_size = 3 * 32 * 32;
	size_t layer0_node_size = 2160 * 1;
	size_t layer1_node_size = 360 * 4;
	size_t layer2_node_size = 60 * 6;
	size_t layer3_node_size = 10 * 8;
	size_t output_node_size = 10;

	// 学習用NET構築
	bb::NeuralNet<> net;
	bb::NeuralNetRealToBinary<>	layer_real2bin(input_node_size, input_node_size * 4, mt());
	bb::NeuralNetBinaryLut6<>	layer_lut0(input_node_size * 4, layer0_node_size, mt());
	bb::NeuralNetBinaryLut6<>	layer_lut1(layer0_node_size, layer1_node_size, mt());
	bb::NeuralNetBinaryLut6<>	layer_lut2(layer1_node_size, layer2_node_size, mt());
	bb::NeuralNetBinaryLut6<>	layer_lut3(layer2_node_size, layer3_node_size, mt());
	bb::NeuralNetBinaryToReal<>	layer_bin2real(layer3_node_size, output_node_size, mt());
	auto last_lut_layer = &layer_lut3;
	net.AddLayer(&layer_real2bin);
	net.AddLayer(&layer_lut0);
	net.AddLayer(&layer_lut1);
	net.AddLayer(&layer_lut2);
	net.AddLayer(&layer_lut3);
	//	net.AddLayer(&layer_bin2real);	// 学習時は bin2real 不要

	// 評価用NET構築(ノードは共有)
	bb::NeuralNet<> net_eva;
	net_eva.AddLayer(&layer_real2bin);
	net_eva.AddLayer(&layer_lut0);
	net_eva.AddLayer(&layer_lut1);
	net_eva.AddLayer(&layer_lut2);
	net_eva.AddLayer(&layer_lut3);
	net_eva.AddLayer(&layer_bin2real);

	// 学習ループ
	int iteration = 0;
	for (int epoc = 0; epoc < epoc_size; ++epoc) {
		// 学習状況評価
		layer_real2bin.InitializeCoeff(1);
		layer_bin2real.InitializeCoeff(1);
		net_eva.SetMuxSize(test_mux_size);
		std::cout << get_time() << "s " << "epoc[" << epoc << "] accuracy : " << CalcAccuracy(net_eva) << std::endl;

		for (size_t x_index = 0; x_index < train_images.size(); x_index += max_batch_size) {
			// 末尾のバッチサイズクリップ
			size_t batch_size = std::min(max_batch_size, train_images.size() - x_index);

			// バッチ学習データの作成
			std::vector< std::vector<float> >	batch_images(train_images.begin() + x_index, train_images.begin() + x_index + batch_size);
			std::vector< std::uint8_t >			batch_labels(train_labels.begin() + x_index, train_labels.begin() + x_index + batch_size);

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
#if 0
			layer_real2bin.InitializeCoeff(1);
			layer_bin2real.InitializeCoeff(1);
			net_eva.SetMuxSize(test_mux_size);
			std::cout << get_time() << "s " << "epoc[" << epoc << "] accuracy : " << CalcAccuracy(net_eva) << std::endl;
#endif

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



int main()
{
	// ファイル読み込み
	std::vector< std::vector<std::uint8_t> > train_images_u8;
	std::vector< std::vector<std::uint8_t> > test_images_u8;

	read_cifer10("data_batch_1.bin", train_labels, train_images_u8);
	read_cifer10("data_batch_2.bin", train_labels, train_images_u8);
	read_cifer10("data_batch_3.bin", train_labels, train_images_u8);
	read_cifer10("data_batch_4.bin", train_labels, train_images_u8);
	read_cifer10("data_batch_5.bin", train_labels, train_images_u8);
	read_cifer10("test_batch.bin", test_labels, test_images_u8);

#ifdef _DEBUG
	std::cout << "!!! DEBUG !!!\n" << std::endl;
	int train_size = 512;
	int test_size  = 256;

	train_labels.resize(train_size);
	train_images_u8.resize(train_size);
	test_labels.resize(test_size);
	test_images_u8.resize(test_size);
#endif

	// 正規化
	train_images = bb::DataTypeConvert<std::uint8_t, float, float>(train_images_u8, 1.0f / 255.0f);
	train_onehot = bb::LabelToOnehot<std::uint8_t, float>(train_labels, 10);
	test_images = bb::DataTypeConvert<std::uint8_t, float, float>(test_images_u8, 1.0f / 255.0f);
	test_onehot = bb::LabelToOnehot<std::uint8_t, float>(test_labels, 10);

	// 表示確認
#if 0
	cv::Mat img(32, 32, CV_32FC3);
	for (int frame = 0; frame < 100; frame++) {
		for (int i = 0; i < 32 * 32; i++) {
			img.at<cv::Vec3f>(i / 32, i % 32)[2] = train_images[frame][(32 * 32) * 0 + i];
			img.at<cv::Vec3f>(i / 32, i % 32)[1] = train_images[frame][(32 * 32) * 1 + i];
			img.at<cv::Vec3f>(i / 32, i % 32)[0] = train_images[frame][(32 * 32) * 2 + i];
		}
		std::cout << (int)train_labels[frame] << std::endl;
		cv::imshow("img", img);
		cv::waitKey();
	}
#endif

	
	//////
#if 1
	// バイナリ6入力LUT版学習実験(重いです)
	RunFlatBinaryLut6(100, 16*8192, -1);
#endif

#if 0
	RunDenseAffineSigmoid(100, 256, 1.0);
#endif

	return 0;
}
