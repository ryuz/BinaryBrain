#include <stdio.h>
#include <vector>
#include <random>
#include <utility>
#include "LutModel.h"
#include "Lut6ModelAvx2.h"


#define	LUT_SIZE		6


#define	INPUT_NUM		28*28
#define	OUTPUT_NUM		10



std::vector<int>	layer_num{ INPUT_NUM, 200, 50, OUTPUT_NUM };
//std::vector<int>	layer_num{INPUT_NUM, 300, 100, 50, OUTPUT_NUM};


std::vector<int> get_randum_num(size_t n, size_t size, std::mt19937& mt)
{
	std::vector<int>	idx(size);

	for (size_t i = 0; i < size; i++) {
		idx[i] = (int)i;
	}

	std::uniform_int_distribution<size_t>	distribution(0, size - 1);
	for (int i = 0; i < n; i++) {
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


// MNIST
inline int mnist_read_word(unsigned char *p)
{
	return (p[0] << 24) + (p[1] << 16) + (p[2] << 8) + (p[3] << 0);
}

std::vector< std::vector<uint8_t> > mnist_read_image(const char* filename)
{
	std::vector< std::vector<uint8_t> >	vec;

	FILE* fp;
	if (fopen_s(&fp, filename, "rb") != 0) {
		return vec;
	}

	unsigned char header[16];
	fread(header, 16, 1, fp);

	int magic = mnist_read_word(&header[0]);
	int num   = mnist_read_word(&header[4]);
	int rows  = mnist_read_word(&header[8]);
	int cols  = mnist_read_word(&header[12]);

	vec.resize(num);
	for (auto& v : vec) {
		v.resize(cols*rows);
		fread(&v[0], cols*rows, 1, fp);
	}
	
	fclose(fp);

	return vec;
}


std::vector< uint8_t> mnist_read_labels(const char* filename)
{
	std::vector<uint8_t>	vec;

	FILE* fp;
	if (fopen_s(&fp, filename, "rb") != 0) {
		return vec;
	}

	unsigned char header[8];
	fread(header, 8, 1, fp);

	int magic = mnist_read_word(&header[0]);
	int num = mnist_read_word(&header[4]);
	vec.resize(num);
	fread(&vec[0], num, 1, fp);
	fclose(fp);

	return vec;
}



void set_input_random(LutNet<LUT_SIZE>& net, std::vector<uint8_t> img, std::mt19937& mt)
{
	std::uniform_int_distribution<int>	distribution(-10, 255 + 10);
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
//		printf("%d / %d\n", ok, n);
	}

	return (float)ok / (float)n;
}



void update(LutNet<LUT_SIZE>& net,
	std::array< std::vector< std::vector<uint8_t> >, 10> image, std::mt19937& mt)
{
	net.Reset();
	for (int i = 0; i < 10; i++) {

	}

}


void update(LutNet<LUT_SIZE>& net,
	std::vector< std::vector<uint8_t> > image, std::vector<uint8_t> label, std::mt19937& mt)
{
	std::uniform_int_distribution<int>	distribution(0, 255);

	net.Reset();

	for (size_t layer = layer_num.size() - 1; layer > 0; layer--) {
		for (auto& lut : net[layer]) {
			lut.ResetScore();
			
			for (size_t i = 0; i < image.size(); i++ ) {
				int th = 127; //  distribution(mt);
				lut.SetReverse(false);
				set_input_th(net, image[i], th);
				net.CalcForward();
				lut.AddScore(net.GetScore(label[i]));

				lut.SetReverse(true);
				set_input_th(net, image[i], th);
				net.CalcForward();
				lut.AddScore(-net.GetScore(label[i]));
				
	//			for (int i = 0; i < 16; i++) {
	//				lut.SetReverse(false);
	//				set_input_random(net, image[i], mt);
	//				net.CalcForward();
	//				lut.AddScore(net.GetScore(label[i]));

	//				lut.SetReverse(true);
	//				set_input_random(net, image[i], mt);
	//				net.CalcForward();
	//				lut.AddScore(-net.GetScore(label[i]));
	//			}
			}
			lut.SetReverse(false);

			lut.Update(mt);
		}
	}
}



int main()
{
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
	int batch_size = 2000;
	auto batch_image = train_image;
	auto batch_label = train_label;
	batch_image.resize(batch_size);
	batch_label.resize(batch_size);

	printf("init\n");

	// LUT構築
	LutNet<LUT_SIZE> net(layer_num);// , mt);
	Lut6Net net6(layer_num);

	// LUTを乱数で初期化
	std::uniform_int_distribution<int>	rand_0_1(0, 1);
	for (size_t layer = 1; layer < layer_num.size(); layer++ ) {
		for (size_t node = 0; node <  layer_num[layer]; node++) {
			for ( int i = 0; i < (1<<LUT_SIZE); i++) {
				bool val = rand_0_1(mt) == 1 ? true : false;
				net[layer][node][i] = val;
				net6[layer].SetLutBit(node, i, val);
			}
		}
	}

	// ランダムに接続
	for (size_t layer = 1; layer < layer_num.size(); layer++) {

		// テーブル作成
		int input_num = layer_num[layer - 1];
		std::uniform_int_distribution<size_t>	rand_con(0, input_num - 1);
		std::vector<int>	idx(input_num);
		for (size_t i = 0; i < input_num; i++) {
			idx[i] = (int)i;
		}

		for (size_t node = 0; node < layer_num[layer]; node++) {
			// シャッフル
			for (int i = 0; i < LUT_SIZE; i++) {
				std::swap(idx[i], idx[rand_con(mt)]);
			}

			// 接続
			for (int i = 0; i < LUT_SIZE; i++) {
				net[layer][node].SetConnection(i, &net[layer - 1][idx[i]]);
				net6[layer].SetInputConnection(node, i, idx[i]);
			}
		}
	}

	printf("start\n");


	printf("%f\n", test_net(net, train_image, train_label));
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
		update(net, batch_image, batch_label, mt);
		
		printf("%f\n", test_net(net, test_image, test_label));
	}
	printf("%f\n", test_net(net, train_image, train_label));

	//	printf("%f\n", test_net(net, test_image, test_label));
//	update(net, train_image, train_label, mt);
//	printf("%f\n", test_net(net, test_image, test_label));
	
	return 0;
}




/*
void calc_forward(std::vector< std::vector< LutModel<LUT_SIZE> > >& lut)
{
	for ( size_t i = 1; i < lut.size(); i++ ) {
		for (auto &l : lut[i]) {
			l.CalcForward();
		}
	}
}

int max_param(std::vector< std::vector< LutModel<LUT_SIZE> > >& lut)
{

}


float eval_lut(
		std::vector< std::vector< LutModel<LUT_SIZE> > >& lut,
		std::vector< std::vector<uint8_t> > image, std::vector<uint8_t> label)
{
	auto l = label.begin();
	for (auto& i : image) {

	}
}



int main()
{
	// MNISTデータ読み込み
	auto train_image = mnist_read_image("train-images-idx3-ubyte");
	auto train_label = mnist_read_labels("train-labels-idx1-ubyte");
	auto test_image  = mnist_read_image("t10k-images-idx3-ubyte");
	auto test_label  = mnist_read_labels("t10k-labels-idx1-ubyte");


	// LUT構築
	std::vector< std::vector< LutModel<LUT_SIZE> > > lut;
	lut.resize(layers);
	for (int i = 0; i < layers; i++) {
		lut[i].resize(layer_num[i]);
	}

	// LUTを乱数で初期化
	std::mt19937						mt(1);
	std::uniform_int_distribution<int>	distribution(0, 1);
	for ( auto& ll : lut ) {
		for (auto& l : ll) {
			for (int i = 0; i < LUT_SIZE; i++) {
				l[i] = distribution(mt) == 0 ? true : false;
			}
		}
	}

	// ランダムに接続
	for (int i = 1; i < layers; i++) {
		auto vec = make_init_vec(lut[i - 1].size());
		for (auto& l : lut[i]) {
			shuffle_vec(mt, vec, LUT_SIZE);
			for (int j = 0; j < LUT_SIZE; j++) {
				l.SetConnection(j, &lut[i-1][vec[j]]);
			}
		}
	}



	return 0;
}

*/