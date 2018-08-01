#include <stdio.h>
#include <vector>
#include <random>
#include <utility>
#include "LutModel.h"

#define	LUT_SIZE		6


#define	INPUT_NUM		28*28
#define	OUTPUT_NUM		10



std::vector<int>	layer_num{INPUT_NUM, 200, 50, OUTPUT_NUM};


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
	//	int th = 127;
	//	for (int th = 0; th < 255; th += 32 ) {
	//		set_input_th(net, img, (uint8_t)th);
	//		net.CalcForward();
	//	}

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

	for (int layer = layer_num.size() - 1; layer > 0; layer--) {
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

	// trainデータ再構築
	/*
	std::array< std::vector< std::vector<uint8_t> >, 10>	tain_data;
	for (size_t i = 0; i < train_image.size(); i++) {
		tain_data[train_label[i]].push_back(train_image[i]);
	}
	*/

	// データ選択用
	std::vector<size_t> train_idx(train_image.size());
	for (size_t i = 0; i < train_image.size(); i++) {
		train_idx[i] = i;
	}
	std::uniform_int_distribution<int>	distribution(0, train_image.size() - 1);
	int batch_size = 2000;
	auto batch_image = train_image;
	auto batch_label = train_label;
	batch_image.resize(batch_size);
	batch_label.resize(batch_size);

	// LUT構築
	LutNet<LUT_SIZE> net(layer_num, mt);
	
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