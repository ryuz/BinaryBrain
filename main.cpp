#include <windows.h>
#include <tchar.h>
#pragma comment(lib, "winmm.lib")

#include <omp.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <utility>
#include "opencv2/opencv.hpp"
#include "mnist_read.h"
#include "ShuffleSet.h"

#include "LutNetBatch.h"
#include "Lut6NetBatchAvx2.h"
#include "Lut6NetBatchAvx2Byte.h"
#include "Lut6NetBatchAvx2Bit.h"


#define	USE_NET0					1
#define	USE_NET1					0

#define	INPUT_NUM					(28*28)
#define	OUTPUT_NUM					10
#define	LUT_SIZE					6

#define	EVA_RATE					2

#define	BATCH_SIZE					(8*1024)

#define UPDATE_FIX_TH_LOOP			1
#define UPDATE_RAND_TH_LOOP			2
#define UPDATE_RAND_PIX_LOOP		2

#define EVALUATION_FIX_TH_LOOP		1
#define EVALUATION_RAND_TH_LOOP		8 //32
#define EVALUATION_RAND_PIX_LOOP	8 //32




//#define UPDATE_GAIN				2.0
//#define UPDATE_GAIN				100.0
#define UPDATE_GAIN					10.0

#define IMG_RAND_TH_MIN			(0   - 127)
#define IMG_RAND_TH_MAX			(255 + 127)
//#define IMG_RAND_TH_MIN				(0)
//#define IMG_RAND_TH_MAX				(254)



//std::vector<int>	net_definition{ INPUT_NUM, 200, 50, OUTPUT_NUM };
//std::vector<int>	net_definition{ INPUT_NUM, 360, 60, OUTPUT_NUM };
//std::vector<int>	net_definition{ INPUT_NUM, 200, 100, 50, OUTPUT_NUM };
//std::vector<int>	net_definition{ INPUT_NUM, 300, 200, 50, OUTPUT_NUM };
//std::vector<int>	net_definition{ INPUT_NUM, 400, 300, 200, 50, OUTPUT_NUM };
//std::vector<int>	net_definition{ INPUT_NUM, 400, 600, 200, 50, OUTPUT_NUM };
//std::vector<int>	net_definition{ INPUT_NUM, 600, 600, 300, 50, OUTPUT_NUM };
//std::vector<int>	net_definition{ INPUT_NUM, 600, 1000, 600, 300, 50, OUTPUT_NUM };
//std::vector<int>	net_definition{ INPUT_NUM, 2000, 1000, 600, 300, 50, OUTPUT_NUM };
//std::vector<int>	net_definition{ INPUT_NUM, 1000, 200, 50, OUTPUT_NUM };
//std::vector<int>	net_definition{ INPUT_NUM, 2160, 360, 60, OUTPUT_NUM };
//std::vector<int>	net_definition{ INPUT_NUM, 1600, 360, 60, OUTPUT_NUM };
//std::vector<int>	net_definition{ INPUT_NUM, 2000, 360, 360, 360, 60, OUTPUT_NUM };
//std::vector<int>	net_definition{ INPUT_NUM, 360*3, 60*3, OUTPUT_NUM*7 };
//std::vector<int>	net_definition{ INPUT_NUM, 2000, 2000, 360 * 3, 60*3, OUTPUT_NUM*3 };
std::vector<int>	net_definition{ INPUT_NUM, 2048, 360 * 3, 60 * 3, OUTPUT_NUM*3};
//std::vector<int>	net_definition{ INPUT_NUM, 360*3, 60*3, OUTPUT_NUM*3 };

//std::vector<int>	layer_num{INPUT_NUM, 4096, 512, 128, 32, OUTPUT_NUM};


void init_net(BinaryNetBatch& net, std::uint64_t seed);
void update_net(BinaryNetBatch& net, std::vector< std::vector<uint8_t> > image, std::vector<uint8_t> label, std::uint64_t seed);

float evaluate_net(BinaryNetBatch& net, std::vector< std::vector<uint8_t> > image, std::vector<uint8_t> label);
double evaluate_net(std::vector<BinaryNetBatch*>& nets, std::vector< std::vector<uint8_t> > image, std::vector<uint8_t> label, std::vector<double>& vec_accuracy);



void WriteRtl(std::ostream& os, BinaryNetData& bnd);


// ネット管理クラス
class ManageNet
{
public:
	std::vector<BinaryNetBatch*>	nets;
	std::mt19937_64					mt;
	std::string						name;

protected:
	DWORD							m_start_time;
	int								m_serial_num = 0;
	double							m_max_accuracy = 0;
	std::vector<double>				m_unit_max_accuracy;

public:
	ManageNet(int net_num=1, std::string name="net", std::uint64_t seed = 1) : nets(net_num), mt(seed), m_unit_max_accuracy(net_num, 0)
	{
		this->name = name;
		m_start_time = timeGetTime();
	}

	BinaryNetBatch* &operator[](int i)
	{
		return nets[i];
	}

	int size(void) { return (int)nets.size(); }


	void WriteLog(double accuracy, std::vector<double> unit_accuracy)
	{
		char fname[64];
		DWORD tm = (timeGetTime() - m_start_time) / 1000;

		// 最大値更新
		if (accuracy > m_max_accuracy) {
			m_max_accuracy = accuracy;
		}

		// 個別ログ
		for (int i = 0; i < size(); i++) {
			auto bnd = nets[i]->ExportData();

			// 精度更新なら保存
			if (unit_accuracy[i] > m_unit_max_accuracy[i]) {
				m_unit_max_accuracy[i] = unit_accuracy[i];

				// RTL保存
				sprintf_s<64>(fname, "%s_%d_rtl_max.v", name.c_str(), i);
				WriteRtl(std::ofstream(fname), bnd);

				// JSON保存
				sprintf_s<64>(fname, "%s_%d_net_max.json", name.c_str(), i);
				std::ofstream ofsJson(fname);
				cereal::JSONOutputArchive o_archive(ofsJson);
				o_archive(bnd);
			}

			{
				// RTL保存
				sprintf_s<64>(fname, "%s_%d_rtl_last.v", name.c_str(), i);
				WriteRtl(std::ofstream(fname), bnd);

				// JSON保存
				sprintf_s<64>(fname, "%s_%d_net_last.json", name.c_str(), i);
				std::ofstream ofsJson(fname);
				cereal::JSONOutputArchive o_archive(ofsJson);
				o_archive(bnd);
			}

			{
				// RTL保存
				sprintf_s<64>(fname, "%s_%d_rtl_%04d.v", name.c_str(), i, m_serial_num);
				WriteRtl(std::ofstream(fname), bnd);

				// JSON保存
				sprintf_s<64>(fname, "%s_%d_net_%04d.json", name.c_str(), i, m_serial_num);
				std::ofstream ofsJson(fname);
				cereal::JSONOutputArchive o_archive(ofsJson);
				o_archive(bnd);
			}


			// ログ
			sprintf_s<64>(fname, "%s_%d_log.txt", name.c_str(), i);
			std::ofstream ofsLog(fname, std::ios::app);
			std::stringstream ss_log;
			ss_log << "[" << m_serial_num << "] " << tm << "s " << name << "(" << i << ") : " << unit_accuracy[i] << " (max : " << m_unit_max_accuracy[i] << ")";
			std::cout << ss_log.str() << std::endl;
			ofsLog << ss_log.str() << std::endl;
		}

		// ログ
		sprintf_s<64>(fname, "%s_log.txt", name.c_str());
		std::ofstream ofsLog(fname, std::ios::app);
		std::stringstream ss_log;
		ss_log << "[" << m_serial_num << "] " << tm << "s " << name << " : " << accuracy << " (max : " << m_max_accuracy << ")";
		std::cout << ss_log.str() << std::endl;
		ofsLog << ss_log.str() << std::endl;

		// グラフ用
		sprintf_s<64>(fname, "%s_accuracy.csv", name.c_str());
		std::ofstream ofsGraph(fname, std::ios::app);
		ofsGraph << accuracy;
		for (int i = 0; i < size(); i++) {
			ofsGraph << "," << unit_accuracy[i];
		}
		ofsGraph << std::endl;

		m_serial_num++;
	}
};



// Binaryネットワークを複数評価するセット
int main()
{
	omp_set_num_threads(6);

	std::mt19937	mt(1);

	// MNISTデータ読み込み
	auto train_image = mnist_read_image("train-images-idx3-ubyte");
	auto train_label = mnist_read_labels("train-labels-idx1-ubyte");
	auto test_image = mnist_read_image("t10k-images-idx3-ubyte");
	auto test_label = mnist_read_labels("t10k-labels-idx1-ubyte");

	// trainデータに10%ほど、ランダム画像を追加
	int add_num = (int)train_image.size() / 10;
	for (int i = 0; i < add_num; i++ ) {
		std::uniform_int_distribution<int> dis(0, 255);
		std::vector<std::uint8_t> img(28 * 28);
		for (auto& v : img) {
			v = dis(mt);
		}
		train_image.push_back(img);
		train_label.push_back(10);
	}

	// ディレクトリを作成
	SYSTEMTIME	systim;
	GetLocalTime(&systim);
	TCHAR	date_buf[64];
	GetDateFormat(LOCALE_USER_DEFAULT, 0, &systim, _T("yyyyMMdd"), date_buf, 64);
	TCHAR	time_buf[64];
	GetTimeFormat(LOCALE_USER_DEFAULT, 0, &systim, _T("HHmmss"), time_buf, 64);
	TCHAR	dir_buf[64];
	_stprintf_s<64>(dir_buf, _T("%s_%s"), date_buf, time_buf);
	::CreateDirectory(dir_buf, NULL);
	::SetCurrentDirectory(dir_buf);


#ifdef _DEBUG
test_image.resize(10);
test_label.resize(10);
#endif


	// データ選択用
	std::vector<size_t> train_idx(train_image.size());
	for (size_t i = 0; i < train_image.size(); i++) {
		train_idx[i] = i;
	}
	std::uniform_int_distribution<int>	distribution(0, (int)train_image.size() - 1);
	int batch_size = BATCH_SIZE;
	auto batch_image = train_image;
	auto batch_label = train_label;
	batch_image.resize(batch_size);
	batch_label.resize(batch_size);

	// ネット構築(暫くは複数種類まわして、結果一致を見ながらバグ取り)
	//	Lut6NetBatchAvx2		net_batch0(net_definition);
	//	Lut6NetBatchAvx2		net_batch1(net_definition);
	//	Lut6NetBatchAvx2Byte	net_batch0(net_definition);

	std::vector<ManageNet> mng_nets;

	mng_nets.push_back(ManageNet(2, "net0", 1));
	Lut6NetBatchAvx2Bit		net00(net_definition);
	Lut6NetBatchAvx2Bit		net01(net_definition);
	Lut6NetBatchAvx2Bit		net02(net_definition);
	Lut6NetBatchAvx2Bit		net03(net_definition);
	mng_nets[0][0] = &net00;
	mng_nets[0][1] = &net01;
	//	mng_nets[0][2] = &net02;
	//	mng_nets[0][3] = &net03;


		// ネットワークを初期化
	for (auto& nets : mng_nets) {
		for (auto& net : nets.nets) {
			init_net(*net, nets.mt());
		}
	}

	// 初期評価
	//	for (auto& nets : mng_nets) {
	//		for ( int i = 0; i < nets.size(); i++ ) {
	//			std::cout << nets.name << "[" << i << "] " << evaluate_net(*nets[i], test_image, test_label) << std::endl;
	//		}
	//	}

double	max_rate = 0;
for (int iteration = 0; iteration < 1000; iteration++) {
	// 定期的に評価して記録
	if ((iteration % EVA_RATE) == 0) {
		for (auto& nets : mng_nets) {
			std::vector<double> unit_accuracy;
			auto accuracy = evaluate_net(nets.nets, test_image, test_label, unit_accuracy);
			nets.WriteLog(accuracy, unit_accuracy);
			//			std::cout << nets.name << "[all] : "  << accuracy << std::endl;
		}
	}

	// 学習実施
	for (auto& nets : mng_nets) {
		for (int i = 0; i < nets.size(); i++) {

			// 学習データ選択
			for (int i = 0; i < batch_size; i++) {
				std::swap(train_idx[i], train_idx[distribution(mt)]);
			}
			for (int i = 0; i < batch_size; i++) {
				batch_image[i] = train_image[train_idx[i]];
				batch_label[i] = train_label[train_idx[i]];
			}


			DWORD tm_b0_s = timeGetTime();
			update_net(*nets[i], batch_image, batch_label, nets.mt());
			DWORD tm_b0_e = timeGetTime();

			std::cout << nets.name << "[" << i << "] : " << (int)(tm_b0_e - tm_b0_s) << " sec" << std::endl;
		}
	}
}

getchar();

return 0;
}



std::array<bool, LUT_SIZE> TestLutBitValidity(BinaryNetBatch& net, int layer, int node)
{
	std::array<bool, LUT_SIZE>	validity;

	std::array<bool, (1 << LUT_SIZE) >	table;
	for (int bit = 0; bit < (1 << LUT_SIZE); bit++ ) {
		table[bit] = net.GetLutBit(layer, node, bit);
	}

	int mask = 1;
	for (int i = 0; i < LUT_SIZE; i++) {
		std::vector<bool>	table0;
		std::vector<bool>	table1;
		table0.reserve(1 << (LUT_SIZE - 1));
		table1.reserve(1 << (LUT_SIZE - 1));
		for (int bit = 0; bit < (1 << LUT_SIZE); bit++) {
			if (bit & mask) {
				table1.push_back(table[bit]);
			}
			else {
				table0.push_back(table[bit]);
			}
		}

		validity[i] = !std::equal(table0.cbegin(), table0.cend(), table1.cbegin());

		mask <<= 1;
	}

	return validity;
}


void evaluate_validity(BinaryNetBatch& net)
{
	int layer_num = net.GetLayerNum();
	for (int layer = 1; layer < layer_num; layer++) {
		int input_n = 0;
		int input_valid = 0;
		int node_num = net.GetNodeNum(layer);
		for (int node = 0; node < node_num; node++) {
			auto vec = TestLutBitValidity(net, layer, node);
			for (auto v : vec) {
				input_valid += v ? 1 : 0;
				input_n++;
			}
		}
		printf("layer[%d]: %d/%d = %f\n", layer, input_valid, input_n, (double)input_valid / input_n);
	}
}


// ネットの初期化
void init_net(BinaryNetBatch& net, std::uint64_t seed)
{
	// 乱数生成
	std::mt19937_64						mt(seed);
	std::uniform_int_distribution<int>	rand_0_1(0, 1);

	// LUT初期化
	int layer_num = net.GetLayerNum();
	for (int layer = 1; layer < layer_num; layer++) {
		int node_num = net.GetNodeNum(layer);
		for (int node = 0; node < node_num; node++) {
			int lut_num = net.GetInputNum(layer, node);
			for (int i = 0; i < (1 << lut_num); i++) {
				bool val = rand_0_1(mt) == 1 ? true : false;
				net.SetLutBit(layer, node, i, val);
			}
		}
	}

	// ランダムに接続
	for (int layer = 1; layer < layer_num; layer++) {
#if 0
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
				net_mono0.SetConnection(layer, node, i, idx[i]);
				net_mono1.SetConnection(layer, node, i, idx[i]);
			}
		}
#else
		// 重複が無いようにランダム結線
		ShuffleSet	ss(net.GetNodeNum(layer - 1), mt());
		for (int node = 0; node < net.GetNodeNum(layer); node++) {
			auto set = ss.GetSet(LUT_SIZE);
			for (int i = 0; i < LUT_SIZE; i++) {
				net.SetConnection(layer, node, i, set[i]);
			}
		}
#endif
	}
}



// 多値画像から乱数で２値画像入力
std::vector<bool> make_input_random(std::vector<uint8_t> img, unsigned int seed)
{
	std::vector<bool> input_vector(img.size());

	std::mt19937						mt(seed);
	std::uniform_int_distribution<int>	distribution(IMG_RAND_TH_MIN, IMG_RAND_TH_MAX);

	for (int i = 0; i < (int)img.size(); i++) {
		input_vector[i] = (img[i] > distribution(mt));
	}

	return input_vector;
}


// 多値画像から固定閾値で２値画像入力
std::vector<bool> make_input_th(std::vector<uint8_t> img, uint8_t th)
{
	std::vector<bool> input_vector(img.size());

	for (int i = 0; i < (int)img.size(); i++) {
		input_vector[i] = (img[i] > th);
	}

	return input_vector;
}



// 出力を評価
double calc_score(int exp, std::vector<bool> out_vec)
{
	double score = 0;
	for (int i = 0; i < (int)out_vec.size(); i++) {
		if ( (i % 10) == exp) {
			score += out_vec[i] ? +1.0 : -1.0;
		}
		else {
			score += out_vec[i] ? -0.1 : +0.1;
//			score += out_vec[i] ? -1.0 : +1.0;
		}
	}
	return score;
}



// ネットを更新(バッチ版)
void update_net(BinaryNetBatch& net, std::vector< std::vector<uint8_t> > image, std::vector<uint8_t> label, std::uint64_t seed)
{
	std::mt19937_64						mt(seed);
	std::uniform_int_distribution<int>	distribution(IMG_RAND_TH_MIN, IMG_RAND_TH_MAX);

	int image_size = (int)image.size();
	int loop_num   = UPDATE_FIX_TH_LOOP + UPDATE_RAND_TH_LOOP + UPDATE_RAND_PIX_LOOP;
	int batch_size = image_size * loop_num;
	net.SetBatchSize(batch_size);

	// 学習データ準備
	int frame = 0;
	for (int i = 0; i < image_size; i++) {
		for (int j = 0; j < UPDATE_FIX_TH_LOOP; j++) {
			net.SetInput(frame++, make_input_th(image[i], 127));
		}
		for (int j = 0; j < UPDATE_RAND_TH_LOOP; j++) {
			net.SetInput(frame++, make_input_th(image[i], distribution(mt)));
		}
		for (int j = 0; j < UPDATE_RAND_PIX_LOOP; j++) {
			net.SetInput(frame++, make_input_random(image[i], mt()));
		}
	}

#if 1
	// 気まぐれにデータをネガポジ反転
	for (int i = 0; i < batch_size; i++) {
		if ( mt() % 2 == 0 ) {
			auto data = net.GetInput(i);
			for (auto& d : data) {
				d = !d;
			}
			net.SetInput(i, data);
		}
	}
#endif

	// 初回計算
	net.CalcForward();
	
	for (int layer = net.GetLayerNum() - 1; layer > 0; layer--) {
		int node_num = net.GetNodeNum(layer);
		for (int node = 0; node < node_num; node++) {
			double  score_val[64] = { 0 };
			int     score_n[64] = { 0 };

			// 集計
			int frame = 0;
			for (int i = 0; i < image_size; i++) {
				for (int j = 0; j < loop_num; j++) {
					int  idx = net.GetInputLutIndex(frame, layer, node);
					score_val[idx] += calc_score(label[i], net.GetOutput(frame));
					score_n[idx]++;
					frame++;
				}
			}

			// 特定ノードを反転させて計算
			net.InvertLut(layer, node);
			net.CalcForward(layer);

			// 集計
			frame = 0;
			for (int i = 0; i < image_size; i++) {
				for (int j = 0; j < loop_num; j++) {
					int  idx = net.GetInputLutIndex(frame, layer, node);
					score_val[idx] -= calc_score(label[i], net.GetOutput(frame));
					score_n[idx]++;
					frame++;
				}
			}

			// 反転を元に戻す
			net.InvertLut(layer, node);
			
			// LUT更新
			std::uniform_real_distribution<double> score_th(-1.0, 0.0);
			for (int i = 0; i < 64; i++) {
				double score = 0.0;
				if (score_n[i] > 0) {
					score = (double)score_val[i] / (double)score_n[i];
				}

		//		if (score * UPDATE_GAIN < score_th(mt)) {
				if ( score <= 0.0 ) {
					net.SetLutBit(layer, node, i, !net.GetLutBit(layer, node, i));
				}
			}

			// 更新箇所以降を再計算
			net.CalcForwardNode(layer, node);
			net.CalcForward(layer);
		}
	}
}




double evaluate_net(std::vector<BinaryNetBatch*>& nets, std::vector< std::vector<uint8_t> > image, std::vector<uint8_t> label, std::vector<double>& vec_accuracy)
{
	static int serial = 0;
	serial++;

	std::mt19937	mt(1);
	std::uniform_int_distribution<int>	distribution(IMG_RAND_TH_MIN, IMG_RAND_TH_MAX);

	int image_size = (int)image.size();
	int loop_num = EVALUATION_FIX_TH_LOOP + EVALUATION_RAND_TH_LOOP + EVALUATION_RAND_PIX_LOOP;
	int batch_size = image_size * loop_num;

	// データ設定
	for (auto& net : nets) {
		net->SetBatchSize(image_size * loop_num);
	}
	int	frame = 0;
	for (int img_idx = 0; img_idx < image_size; img_idx++) {
		// 固定閾置
		for (int i = 0; i < EVALUATION_FIX_TH_LOOP; i++) {
			auto input_vec = make_input_th(image[img_idx], 127);
			for (auto& net : nets) {
				net->SetInput(frame, input_vec);
			}
			frame++;
		}

		// フレーム単位乱数閾置
		for (int i = 0; i < EVALUATION_RAND_TH_LOOP; i++) {
			auto input_vec = make_input_th(image[img_idx], mt());
			for (auto& net : nets) {
				net->SetInput(frame, input_vec);
			}
			frame++;
		}

		// ピクセル単位乱数閾置
		for (int i = 0; i < EVALUATION_RAND_PIX_LOOP; i++) {
			auto input_vec = make_input_random(image[img_idx], mt());
			for (auto& net : nets) {
				net->SetInput(frame, input_vec);
			}
			frame++;
		}	
	}

	// 気まぐれにデータをネガポジ反転
	for (auto& net : nets) {
		for (int i = 0; i < batch_size; i++) {
			if (mt() % 2 == 0) {
				auto data = net->GetInput(i);
				for (auto& d : data) {
					d = !d;
				}
				net->SetInput(i, data);
			}
		}
	}

	// 計算
	for (auto& net : nets) {
		net->CalcForward();
	}

	// 個別評価
	vec_accuracy.clear();
	for (auto& net : nets) {
		int		n = 0;
		int		ok = 0;
		frame = 0;
		for (int img_idx = 0; img_idx < image_size; img_idx++) {
			std::vector<int> count(OUTPUT_NUM, 0);
			for (int i = 0; i < loop_num; i++) {
				auto& out_vec = net->GetOutput(frame);
				for (int j = 0; j < (int)out_vec.size(); j++) {
					count[j % OUTPUT_NUM] += out_vec[j] ? 1 : 0;
				}
				frame++;
			}

			auto	max_it = std::max_element(count.begin(), count.end());
			uint8_t max_idx = (uint8_t)std::distance(count.begin(), max_it);
			if (max_idx == label[img_idx]) {
				ok++;
			}
			n++;
		}

		vec_accuracy.push_back((double)ok / (double)n);
	}


	int error_count[10][10] = { 0 };

	// 総合評価
	int		n = 0;
	int		ok = 0;
	frame = 0;
	for (int img_idx = 0; img_idx < image_size; img_idx++) {
		std::vector<int> count(OUTPUT_NUM, 0);
		for (int i = 0; i < loop_num; i++) {
			for (auto& net : nets) {
				auto& out_vec = net->GetOutput(frame);
				for (int j = 0; j < (int)out_vec.size(); j++) {
					count[j % OUTPUT_NUM] += out_vec[j] ? 1 : 0;
				}
			}
			frame++;
		}

		auto	max_it = std::max_element(count.begin(), count.end());
		uint8_t max_idx = (uint8_t)std::distance(count.begin(), max_it);
		if (max_idx == label[img_idx]) {
			ok++;
		}
		n++;

		error_count[label[img_idx]][max_idx]++;

#if 0
		if (serial > 4 && label[img_idx] != max_idx) {
			printf("ok:%d ng:%d\n", label[img_idx], max_idx);

			cv::Mat img(28, 28, CV_8UC1);
			auto input_vector = nets[0]->GetInput(img_idx * loop_num + 0);
			for (int i = 0; i < input_vector.size(); i++) {
				img.data[i] = input_vector[i] ? 255 : 0;
			}
			cv::imshow("img0", img);

			input_vector = nets[0]->GetInput(img_idx * loop_num + 1);
			for (int i = 0; i < input_vector.size(); i++) {
				img.data[i] = input_vector[i] ? 255 : 0;
			}
			cv::imshow("img32", img);
			cv::waitKey();
		}
#endif
	}

	// LUT使用率チェック
	if (0) {
		for (auto& net : nets) {
			int layer_num = net->GetLayerNum();
			for (int layer = 1; layer < layer_num; layer++) {
				int  hist[65] = { 0 };

				int node_num = net->GetNodeNum(layer);
				for (int node = 0; node < node_num; node++) {
					int  look_n[64] = { 0 };
					for (int frame = 0; frame < batch_size; frame++) {
						int  idx = net->GetInputLutIndex(frame, layer, node);
						look_n[idx]++;
					}

					int nn = 0;
					for (int i = 0; i < 64; i++) {
						nn += look_n[i] > 0 ? 1 : 0;
					}
					hist[nn]++;
				}

				printf("layer%d\n", layer);
				for (int i = 0; i < 65; i++) {
					printf("%d:%d\n", i, hist[i]);
				}
			}
		}
	}

	std::ofstream ofs("err_matrix.txt");
	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 10; j++) {
			ofs << error_count[i][j] << "\t";
		}
		ofs << std::endl;
	}
	
	/*
	for (auto& net : nets) {
		evaluate_validity(*net);
	}
	*/

	return (double)ok / (double)n;
}



float evaluate_net(BinaryNetBatch& net, std::vector< std::vector<uint8_t> > image, std::vector<uint8_t> label)
{

	std::mt19937	mt(1);
	std::uniform_int_distribution<int>	distribution(IMG_RAND_TH_MIN, IMG_RAND_TH_MAX);

	int image_size = (int)image.size();

	net.SetBatchSize(image_size * 64);

	int out_layer = net.GetLayerNum() - 1;
	int out_num = net.GetNodeNum(out_layer);
	int	frame = 0;

	// データ設定
	for (int img_idx = 0; img_idx < image_size; img_idx++) {
		for (int i = 0; i < 32; i++) {
			net.SetInput(frame++, make_input_random(image[img_idx], mt()));
			net.SetInput(frame++, make_input_th(image[img_idx], distribution(mt)));
		}
	}

	// 計算
	net.CalcForward();

	// 評価
	int		n = 0;
	int		ok = 0;

	frame = 0;
	for (int img_idx = 0; img_idx < image_size; img_idx++) {
		std::vector<int> count(10, 0);
		for (int i = 0; i < 64; i++) {
			for (int j = 0; j < out_num; j++) {
				count[j % 10] += net.GetValue(frame, out_layer, j) ? 1 : 0;
			}
			frame++;
		}

		auto	max_it = std::max_element(count.begin(), count.end());
		uint8_t max_idx = (uint8_t)std::distance(count.begin(), max_it);
		if (max_idx == label[img_idx]) {
			ok++;
		}
		n++;
	}
	
	return (float)ok / (float)n;
}

