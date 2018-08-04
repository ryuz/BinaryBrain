#include <windows.h>
#pragma comment(lib, "winmm.lib")

#include <stdio.h>
#include <vector>
#include <random>
#include <utility>
#include "mnist_read.h"
#include "LutModel.h"
#include "Lut6ModelAvx2.h"


#define	LUT_SIZE		6


#define	INPUT_NUM		28*28
#define	OUTPUT_NUM		10



//std::vector<int>	layer_num{ INPUT_NUM, 200, 50, OUTPUT_NUM };
std::vector<int>	layer_num{INPUT_NUM, 4096, 512, 128, 32, OUTPUT_NUM};


#if 0
std::vector<int> get_randum_num(size_t n, size_t size, std::mt19937& mt)
{
	std::vector<int>	idx(size);

	for (size_t i = 0; i < size; i++) {
		idx[i] = (int)i;
	}

	std::uniform_int_distribution<size_t>	distribution(0, size - 1);
	for (int i = 0; i < (int)n; i++) {
		std::swap(idx[i], idx[distribution(mt)]);
	}

	idx.resize(n);

	return idx;
}


template<class T>
std::vector<T> get_randum_data(std::vector<T> data, size_t n, std::mt19937& mt)
{
	n = std::min(n, data.size());

	std::vector<int>	idx(data.size());
	for (size_t i = 0; i < data.size(); i++) {
		idx[i] = i;
	}

	std::uniform_int_distribution<int>	distribution(0, data.size() - 1);
	for (int i = 0; i < n; i++) {
		std::swap(idx[i], idx[distribution(mt)]);
	}

	std::vector<int>	vec(n);
	for (int i = 0; i < n; i++) {
		vec[i] = data[idx[i]];
	}

	return vec;
}
#endif





// ---------------------------------------------
//  data
// ---------------------------------------------

std::vector<bool> make_input_random(std::vector<uint8_t> img, unsigned int seed)
{
	std::vector<bool> input_vector(img.size());

	std::mt19937						mt(seed);
	std::uniform_int_distribution<int>	distribution(-10, 255 + 10);

	for (int i = 0; i < (int)img.size(); i++) {
		input_vector[i] = (img[i] > distribution(mt));
	}

	return input_vector;
}


std::vector<bool> make_input_th(std::vector<uint8_t> img, uint8_t th)
{
	std::vector<bool> input_vector(img.size());

	for (int i = 0; i < (int)img.size(); i++) {
		input_vector[i] = (img[i] > th);
	}

	return input_vector;
}


#if 0

void set_input_random(LutNet<LUT_SIZE>& net, std::vector<uint8_t> img, std::mt19937& mt)
{
//	std::uniform_int_distribution<int>	distribution(10, 255 + 10);

	std::uniform_int_distribution<int>	distribution(10, 254-10);
	auto& in = net.Input();
	auto it = img.begin();
	for (auto& lut : in) {
		int v = (int)*it++;
		int	r = distribution(mt);
		lut.SetValue( v > r );
	}
}



void set_input_th(LutNet<LUT_SIZE>& net, std::vector<uint8_t> img, uint8_t th)
{
	auto& in = net.Input();
	auto it = img.begin();
	for (auto& i : in) {
		i.SetValue(*it++ > th);
	}
}

#endif


float test_net2(BinaryNet& net,
	std::vector< std::vector<uint8_t> > image, std::vector<uint8_t> label)
{
	int		n = 0;
	int		ok = 0;

	std::mt19937	mt(1);
	
	int out_layer = net.GetLayerNum() - 1;
	auto label_it = label.begin();
	for (auto& img : image) {

		std::vector<int> count(10, 0);
		for (int i = 0; i < 16; i++) {
			auto in_vec = make_input_random(img, mt());
			net.SetInput(in_vec);
			net.CalcForward();

			for ( int j = 0; j < 10; j++ ) {
				count[j] += net.GetValue(out_layer, j) ? 1 : 0;
			}
		}

		auto	max_it = std::max_element(count.begin(), count.end());
		uint8_t max_idx = (uint8_t)std::distance(count.begin(), max_it);
		uint8_t label = *label_it++;
		if (max_idx == label) {
			ok++;
		}
		n++;
	}
	
	return (float)ok / (float)n;
}


#if 0
float test_net(LutNet<LUT_SIZE>& net, 
	std::vector< std::vector<uint8_t> > image, std::vector<uint8_t> label)
{
	int		n = 0;
	int		ok = 0;
	
	std::mt19937	mt(1);

	auto label_it = label.begin();
	for (auto& img : image) {
		net.Reset();

		for ( int i = 0; i < 16; i++) {
			set_input_random(net, img, mt);
			net.CalcForward();
		}

		auto	max_it = std::max_element(net.count.begin(), net.count.end());
		uint8_t max_idx = (uint8_t)std::distance(net.count.begin(), max_it);
		uint8_t label = *label_it++;
		if ( max_idx == label ) {
			ok++;
		}
		n++;
	}

	return (float)ok / (float)n;
}
#endif



double calc_score(int exp, std::vector<bool> out_vec)
{
	double score = 0;
	for (int i = 0; i < (int)out_vec.size(); i++) {
		if (i == exp) {
			score += out_vec[i] ? +20.0 : -20.0;
		}
		else {
			score += out_vec[i] ? -1.0 : +1.0;
		}
	}
	return score;
}

void update2(BinaryNet& net, std::vector< std::vector<uint8_t> > image, std::vector<uint8_t> label, std::mt19937& mt)
{
	std::uniform_int_distribution<int>	distribution(0 + 16, 254 - 16);

	for ( int layer = net.GetLayerNum() - 1; layer > 0; layer-- ) {
		int node_num = net.GetNodeNum(layer);
		for (int node = 0; node < node_num; node++ ) {
			double  score_val[64] = { 0 };
			int     score_n[64] = { 0 };

			for ( int i = 0; i < (int)image.size(); i++) {
#if 1			
				for (int j = 0; j < 1; j++) {
					//		int th = 127; //  distribution(mt);
					int th = distribution(mt);
					net.SetInput(make_input_th(image[i], th));
					net.CalcForward();
					int  idx     = net.GetInputLutIndex(layer, node);
					score_val[idx] += calc_score(label[i], net.GetOutput());
					score_n[idx]++;

					net.InvertLut(layer, node);
					net.CalcForward(layer);
					score_val[idx] -= calc_score(label[i], net.GetOutput());
					score_n[idx]++;
					net.InvertLut(layer, node);
				}
#endif

#if 0
				for (int j = 0; j < 1; j++) {
					net.SetInput(make_input_random(image[i], mt()));
					net.CalcForward();
					int  idx = net.GetInputLutIndex(layer, node);
					score_val[idx] += calc_score(label[i], net.GetOutput());
					score_n[idx]++;

					net.InvertLut(layer, node);
					net.CalcForward(layer);
					score_val[idx] -= calc_score(label[i], net.GetOutput());
					score_n[idx]++;
					net.InvertLut(layer, node);
				}
#endif
			}

			std::uniform_int_distribution<int>	distribution(-20, 0);
			int th = distribution(mt);
			for ( int i = 0; i < 64; i++ ) {
				double score = score_val[i] / (double)score_n[i];
				if ( score * 10 < th ) {
					net.SetLutBit(layer, node, i, !net.GetLutBit(layer, node, i));
				}
			}

//			lut.Update(mt);
		}
	}


}


#if 0
void update(LutNet<LUT_SIZE>& net,
	std::vector< std::vector<uint8_t> > image, std::vector<uint8_t> label, std::mt19937& mt)
{
	std::uniform_int_distribution<int>	distribution(0+16, 254-16);

	net.Reset();

	for (size_t layer = layer_num.size() - 1; layer > 0; layer--) {
		for (auto& lut : net[layer]) {
			
			lut.ResetScore();

			for (size_t i = 0; i < image.size(); i++ ) {
#if 1			
				for (int j = 0; j < 1; j++) {
			//		int th = 127; //  distribution(mt);
					int th = distribution(mt);
					lut.SetReverse(false);
					set_input_th(net, image[i], th);
					net.CalcForward();

					lut.AddScore(net.GetScore(label[i]));

					lut.SetReverse(true);
			//		set_input_th(net, image[i], th);
					net.CalcForward();
					lut.AddScore(-net.GetScore(label[i]));
				}
#endif

#if 1
				for (int j = 0; j < 1; j++) {
					lut.SetReverse(false);
					set_input_random(net, image[i], mt);
					net.CalcForward();
					lut.AddScore(net.GetScore(label[i]));

					lut.SetReverse(true);
				//	set_input_random(net, image[i], mt);
					net.CalcForward();
					lut.AddScore(-net.GetScore(label[i]));
				}
#endif
			}
			lut.SetReverse(false);

			lut.Update(mt);
		}
	}
}


void test01(void)
{
	std::vector<int>	layers{6, 1};
	LutNet<LUT_SIZE>	net(layers);
	Lut6Net				net6(layers);

	for (int i = 0; i < 6; i++) {
		net.SetConnection(1, 0, i, i);
		net6.SetConnection(1, 0, i, i);
	}

	for (int i = 0; i < 64; i++) {
		net.SetLutBit(1, 0, i, i==35);
		net6.SetLutBit(1, 0, i, i==35);
	}

	for (int i = 0; i < 6; i++) {
		net.SetValue(0, i, (35 >> i) & 1);
		net6.SetValue(0, i, (35 >> i) & 1);
	}

	net6.CalcForward();
	net.CalcForward();
}
#endif


#include <omp.h>

int main()
{
	omp_set_num_threads(4);
//	test01();

	std::mt19937	mt(1);

	// MNISTデータ読み込み
	auto train_image = mnist_read_image("train-images-idx3-ubyte");
	auto train_label = mnist_read_labels("train-labels-idx1-ubyte");
	auto test_image = mnist_read_image("t10k-images-idx3-ubyte");
	auto test_label = mnist_read_labels("t10k-labels-idx1-ubyte");

	// データ選択用
	std::vector<size_t> train_idx(train_image.size());
	for (size_t i = 0; i < train_image.size(); i++) {
		train_idx[i] = i;
	}
	std::uniform_int_distribution<int>	distribution(0, (int)train_image.size() - 1);
	int batch_size = 100;
	auto batch_image = train_image;
	auto batch_label = train_label;
	batch_image.resize(batch_size);
	batch_label.resize(batch_size);

	printf("init\n");

	// LUT構築
	LutNet<LUT_SIZE>	net(layer_num);
	Lut6Net				net6(layer_num);

	// LUTを乱数で初期化
	std::uniform_int_distribution<int>	rand_0_1(0, 1);
	for (int layer = 1; layer < (int)layer_num.size(); layer++ ) {
		for (int node = 0; node < layer_num[layer]; node++) {
			for ( int i = 0; i < (1<<LUT_SIZE); i++) {
				bool val = rand_0_1(mt) == 1 ? true : false;
				net.SetLutBit(layer, node, i, val);
				net6.SetLutBit(layer, node, i, val);
			}
		}
	}

	// ランダムに接続
	for (int layer = 1; layer < (int)layer_num.size(); layer++) {

		// テーブル作成
		int input_num = layer_num[layer - 1];
		std::uniform_int_distribution<size_t>	rand_con(0, input_num - 1);
		std::vector<int>	idx(input_num);
		for (int i = 0; i < input_num; i++) {
			idx[i] = i;
		}

		for (int node = 0; node < layer_num[layer]; node++) {
			// シャッフル
			for (int i = 0; i < LUT_SIZE; i++) {
				std::swap(idx[i], idx[rand_con(mt)]);
			}

			// 接続
			for (int i = 0; i < LUT_SIZE; i++) {
				net.SetConnection(layer, node, i, idx[i]);
				net6.SetConnection(layer, node, i, idx[i]);
			}
		}
	}



	printf("start\n");
	

	std::mt19937	mt2(2);
	std::mt19937	mt6(2);

	for (int x = 0; x < 1000; x++) {
		// 学習データ選択
		for (int i = 0; i < batch_size; i++) {
			std::swap(train_idx[i], train_idx[distribution(mt)]);
		}
		for (int i = 0; i < batch_size; i++) {
			batch_image[i] = train_image[train_idx[i]];
			batch_label[i] = train_label[train_idx[i]];
		}
		
//		printf("%f\n", test_net(net, train_image, train_label));
		
		DWORD tm0_s = timeGetTime();
	//	update2(net, batch_image, batch_label, mt2);
		DWORD tm0_e = timeGetTime();
		printf("net0:%d[ms]\n", (int)(tm0_e - tm0_s));

		DWORD tm1_s = timeGetTime();
		update2(net6, batch_image, batch_label, mt6);
		DWORD tm1_e = timeGetTime();
		printf("net1:%d[ms]\n", (int)(tm1_e - tm1_s));

		if (x % 2 == 0) {
	//		printf("ref0:%f\n", test_net2(net, test_image, test_label));
			printf("net6:%f\n", test_net2(net6, test_image, test_label));
		}
	}

	//	printf("%f\n", test_net(net, test_image, test_label));
//	update(net, train_image, train_label, mt);
//	printf("%f\n", test_net(net, test_image, test_label));
	
	return 0;
}


