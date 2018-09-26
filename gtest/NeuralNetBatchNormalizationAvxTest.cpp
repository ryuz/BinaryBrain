#include <stdio.h>
#include <iostream>
#include "gtest/gtest.h"
#include "bb/NeuralNetBatchNormalizationAvx.h"
#include "bb/NeuralNetBatchNormalization.h"



template <typename T=float>
class SimpleBatchNorm
{
public:
	SimpleBatchNorm(int n) : n(n), x(n), y(n), dx(n), dy(n)//, xn(n), xc(n), dxn(n), dxc(n)
	{
	}
		
	int				n;
	
	std::vector<T>	x;
	std::vector<T>	y;

	T				mean;
	T				std;
	T				var;
	T				gamma = 1;
	T				beta = 0;

	std::vector<T>	dx;
	std::vector<T>	dy;
	
	T				dmean;
	T				dstd;
	T				dvar;
	T				dgamma;
	T				dbeta;

#if 0
	std::vector<T>	xn;
	std::vector<T>	xc;
	std::vector<T>	dxn;
	std::vector<T>	dxc;

	// オリジナル
	void Forward0(void)
	{
		// 平均
		mean = 0;
		for (int i = 0; i < n; ++i) {
			mean += x[i];
		}
		mean /= (T)n;

		// 平均を引く
		for (int i = 0; i < n; ++i) {
			xc[i] = x[i] - mean;
		}

		// 分散
		var = 0;
		for (int i = 0; i < n; ++i) {
			var += xc[i] * xc[i];
		}
		var /= (T)n;

		// 偏差
		std = sqrt(var + (T)10e-7);

		// 正規化
		for (int i = 0; i < n; ++i) {
			xn[i] = xc[i] / std;
		}

		// シフト
		for (int i = 0; i < n; ++i) {
			y[i] = xn[i] * gamma + beta;
		}
	}

	// mean/var一括 (step順)
	void Forward1(void)
	{
		// 平均/分散一括
		mean = 0;
		var = 0;
		for (int i = 0; i < n; ++i) {
			mean += x[i];
			var += x[i] * x[i];
		}
		mean /= (T)n;
		var = (var / (T)n) - (mean * mean);
		std = sqrt(var + (T)10e-7);

		// 平均を引く
		for (int i = 0; i < n; ++i) {
			xc[i] = x[i] - mean;
		}

		// 正規化
		for (int i = 0; i < n; ++i) {
			xn[i] = xc[i] / std;
		}
			// シフト
		for (int i = 0; i < n; ++i) {
			y[i] = xn[i] * gamma + beta;
		}
	}

	// mean/var一括 (ループ最適化)
	void Forward2(void)
	{
		// 平均/分散一括
		mean = 0;
		var = 0;
		for (int i = 0; i < n; ++i) {
			mean += x[i];
			var  += x[i] * x[i];
		}
		mean /= (T)n;
		var = (var / (T)n) - (mean * mean);
		std = sqrt(var + (T)10e-7);

		for (int i = 0; i < n; ++i) {
			// 平均を引く
			xc[i] = x[i] - mean;

			// 正規化
			xn[i] = xc[i] / std;

			// シフト
			y[i] = xn[i] * gamma + beta;
		}
	}
#endif

	// mean/var一括 (中間変数未使用)
	void Forward(void)
	{
		// 平均/分散一括
		mean = 0;
		var = 0;
		for (int i = 0; i < n; ++i) {
			mean += x[i];
			var += x[i] * x[i];
		}
		mean /= (T)n;
		var = (var / (T)n) - (mean * mean);
		std = sqrt(var + (T)10e-7);

		for (int i = 0; i < n; ++i) {
			// 平均を引く
			T _xc = x[i] - mean;

			// 正規化
			T _xn = _xc / std;

			// シフト
			y[i] = _xn * gamma + beta;
		}
	}

#if 0
	// オリジナル
	void Backward0(void)
	{
		dbeta = 0;
		for (int i = 0; i < n; ++i) {
			dbeta += dy[i];
		}

		dgamma = 0;
		for (int i = 0; i < n; ++i) {
			dgamma += xn[i] * dy[i];
		}

		for (int i = 0; i < n; ++i) {
			dxn[i] = dy[i] * gamma;
		}

		for (int i = 0; i < n; ++i) {
			dxc[i] = dxn[i] / std;
		}

		dstd = 0;
		for (int i = 0; i < n; ++i) {
			dstd += -(dxn[i] * xc[i]) / (std * std);
		}

		dvar = (T)0.5 * dstd / std;

		for (int i = 0; i < n; ++i) {
			dxc[i] = dxc[i] + (((T)2.0 / (T)n) * dvar * xc[i]);
		}

		dmean = 0;
		for (int i = 0; i < n; ++i) {
			dmean += dxc[i];
		}

		for (int i = 0; i < n; ++i) {
			dx[i] = dxc[i] - (dmean / (T)n);
		}
	}


	// mean/var一括 (step順)
	void Backward1(void)
	{
		dbeta = 0;
		for (int i = 0; i < n; ++i) {
			dbeta += dy[i];
		}

		dgamma = 0;
		for (int i = 0; i < n; ++i) {
			dgamma += xn[i] * dy[i];
		}

		for (int i = 0; i < n; ++i) {
			dxn[i] = dy[i] * gamma;
		}

		for (int i = 0; i < n; ++i) {
			dxc[i] = dxn[i] / std;
		}

		dstd = 0;
		for (int i = 0; i < n; ++i) {
			dstd += -(dxn[i] * xc[i]) / (std * std);
		}

		dvar = (T)0.5 * dstd / std;

		T dmeanx = 0;
		for (int i = 0; i < n; ++i) {
			dmeanx += (-dxc[i]);
		}
		dmean = dmeanx + (2.0f * mean * -dvar);
		dmean /= (T)n;

		for (int i = 0; i < n; ++i) {
			dx[i] = dxc[i] + dmean + (((T)2 / (T)n) * x[i] * dvar);
		}
	}


	// mean/var一括 (ループ最適化)
	void Backward2(void)
	{
		dbeta = 0;
		dgamma = 0;
		dstd = 0;
		T dmeanx = 0;
		for (int i = 0; i < n; ++i) {
			dbeta  += dy[i];
			dgamma += xn[i] * dy[i];
			dxn[i] = dy[i] * gamma;
			dxc[i] = dxn[i] / std;
			dstd += -(dxn[i] * xc[i]) / (std * std);
			dmeanx += (-dxc[i]);
		}

		dvar = (T)0.5 * dstd / std;
		dmean = dmeanx + (2.0f * mean * -dvar);
		dmean /= (T)n;

		for (int i = 0; i < n; ++i) {
			dx[i] = dxc[i] + dmean + (((T)2 / (T)n) * x[i] * dvar);
		}
	}
#endif

	// 中間変数未使用 (step順)
	void Backward(void)
	{
		dbeta = 0;
		dgamma = 0;
		dstd = 0;
		T dmeanx = 0;
		for (int i = 0; i < n; ++i) {
			T _xc = (x[i] - mean);
			T _xn = _xc / std;

			dbeta += dy[i];
			dgamma += _xn * dy[i];
			T _dxn = dy[i] * gamma;
			T _dxc = _dxn / std;
			dstd += -(_dxn * _xc) / (std * std);
			dmeanx += (-_dxc);
		}

		dvar = (T)0.5 * dstd / std;
		dmean = dmeanx + (2.0f * mean * -dvar);
		dmean /= (T)n;

		for (int i = 0; i < n; ++i) {
			T _dxn = dy[i] * gamma;
			T _dxc = _dxn / std;

			dx[i] = _dxc + dmean + (((T)2 / (T)n) * x[i] * dvar);
		}
	}



};









inline void testSetupLayerBuffer(bb::NeuralNetLayer<>& net)
{
	net.SetInputSignalBuffer (net.CreateInputSignalBuffer());
	net.SetInputErrorBuffer (net.CreateInputErrorBuffer());
	net.SetOutputSignalBuffer(net.CreateOutputSignalBuffer());
	net.SetOutputErrorBuffer(net.CreateOutputErrorBuffer());
}


TEST(NeuralNetBatchNormalizationAvxTest, testBatchNormalization)
{
	bb::NeuralNetBatchNormalization<> batch_norm(2);
	batch_norm.SetBatchSize(8);
	testSetupLayerBuffer(batch_norm);
	
	SimpleBatchNorm<> exp_norm0(8);
	SimpleBatchNorm<> exp_norm1(8);


	auto in_sig = batch_norm.GetInputSignalBuffer();
	auto out_sig = batch_norm.GetOutputSignalBuffer();
	in_sig.SetReal(0, 0, 1);
	in_sig.SetReal(1, 0, 2);
	in_sig.SetReal(2, 0, 3);
	in_sig.SetReal(3, 0, 4);
	in_sig.SetReal(4, 0, 5);
	in_sig.SetReal(5, 0, 6);
	in_sig.SetReal(6, 0, 7);
	in_sig.SetReal(7, 0, 8);
	in_sig.SetReal(0, 1, 10);
	in_sig.SetReal(1, 1, 30);
	in_sig.SetReal(2, 1, 20);
	in_sig.SetReal(3, 1, 15);
	in_sig.SetReal(4, 1, 11);
	in_sig.SetReal(5, 1, 34);
	in_sig.SetReal(6, 1, 27);
	in_sig.SetReal(7, 1, 16);

	for (int i = 0; i < 8; i++) {
		exp_norm0.x[i] = in_sig.GetReal(i, 0);
		exp_norm1.x[i] = in_sig.GetReal(i, 1);
	}
	

	batch_norm.Forward(true);
//	batch_norm.Forward(false);

	exp_norm0.Forward();
	exp_norm1.Forward();


	/*
	[-1.52752510, -1.23359570],
	[-1.09108940, +1.14442010],
	[-0.65465360, -0.04458780],
	[-0.21821786, -0.63909180],
	[+0.21821786, -1.11469500],
	[+0.65465360, +1.62002340],
	[+1.09108940, +0.78771776],
	[+1.52752510, -0.52019095]
	*/

#if 0
	std::cout << out_sig.GetReal(0, 0) << std::endl;
	std::cout << out_sig.GetReal(1, 0) << std::endl;
	std::cout << out_sig.GetReal(2, 0) << std::endl;
	std::cout << out_sig.GetReal(3, 0) << std::endl;
	std::cout << out_sig.GetReal(4, 0) << std::endl;
	std::cout << out_sig.GetReal(5, 0) << std::endl;
	std::cout << out_sig.GetReal(6, 0) << std::endl;
	std::cout << out_sig.GetReal(7, 0) << std::endl;
#endif

	for (int i = 0; i < 8; i++) {
		EXPECT_TRUE(abs(out_sig.GetReal(i, 0) - exp_norm0.y[i]) < 0.000001);
		EXPECT_TRUE(abs(out_sig.GetReal(i, 1) - exp_norm1.y[i]) < 0.000001);
	}

	// _mm256_rsqrt_ps を使っているので精度は悪い
	EXPECT_TRUE(abs(out_sig.GetReal(0, 0) - -1.52752510) < 0.000001);
	EXPECT_TRUE(abs(out_sig.GetReal(1, 0) - -1.09108940) < 0.000001);
	EXPECT_TRUE(abs(out_sig.GetReal(2, 0) - -0.65465360) < 0.000001);
	EXPECT_TRUE(abs(out_sig.GetReal(3, 0) - -0.21821786) < 0.000001);
	EXPECT_TRUE(abs(out_sig.GetReal(4, 0) - +0.21821786) < 0.000001);
	EXPECT_TRUE(abs(out_sig.GetReal(5, 0) - +0.65465360) < 0.000001);
	EXPECT_TRUE(abs(out_sig.GetReal(6, 0) - +1.09108940) < 0.000001);
	EXPECT_TRUE(abs(out_sig.GetReal(7, 0) - +1.52752510) < 0.000001);

	EXPECT_TRUE(abs(out_sig.GetReal(0, 1) - -1.23359570) < 0.000001);
	EXPECT_TRUE(abs(out_sig.GetReal(1, 1) - +1.14442010) < 0.000001);
	EXPECT_TRUE(abs(out_sig.GetReal(2, 1) - -0.04458780) < 0.000001);
	EXPECT_TRUE(abs(out_sig.GetReal(3, 1) - -0.63909180) < 0.000001);
	EXPECT_TRUE(abs(out_sig.GetReal(4, 1) - -1.11469500) < 0.000001);
	EXPECT_TRUE(abs(out_sig.GetReal(5, 1) - +1.62002340) < 0.000001);
	EXPECT_TRUE(abs(out_sig.GetReal(6, 1) - +0.78771776) < 0.000001);
	EXPECT_TRUE(abs(out_sig.GetReal(7, 1) - -0.52019095) < 0.000001);


#if 1
	auto out_err = batch_norm.GetOutputErrorBuffer();
	auto in_err = batch_norm.GetInputErrorBuffer();
	
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 8; j++) {
			out_err.SetReal(j, i, out_sig.GetReal(j, i));
		}
	}

	out_err.SetReal(0, 0, 8);
	out_err.SetReal(1, 0, 6);
	out_err.SetReal(2, 0, 3);
	out_err.SetReal(3, 0, 4);
	out_err.SetReal(4, 0, 5);
	out_err.SetReal(5, 0, 4);
	out_err.SetReal(6, 0, 6);
	out_err.SetReal(7, 0, 1);
	out_err.SetReal(0, 1, 20);
	out_err.SetReal(1, 1, 70);
	out_err.SetReal(2, 1, 40);
	out_err.SetReal(3, 1, 15);
	out_err.SetReal(4, 1, 31);
	out_err.SetReal(5, 1, 54);
	out_err.SetReal(6, 1, 37);
	out_err.SetReal(7, 1, 26);

	for (int i = 0; i < 8; i++) {
		exp_norm0.dy[i] = out_err.GetReal(i, 0);
		exp_norm1.dy[i] = out_err.GetReal(i, 1);
	}

	batch_norm.Backward();

	exp_norm0.Backward();
	exp_norm1.Backward();


#if 0
	std::cout << in_err.GetReal(0, 0) << std::endl;
	std::cout << in_err.GetReal(1, 0) << std::endl;
	std::cout << in_err.GetReal(2, 0) << std::endl;
	std::cout << in_err.GetReal(3, 0) << std::endl;
	std::cout << in_err.GetReal(4, 0) << std::endl;
	std::cout << in_err.GetReal(5, 0) << std::endl;
	std::cout << in_err.GetReal(6, 0) << std::endl;
	std::cout << in_err.GetReal(7, 0) << std::endl;
#endif

	/*
		[+0.65465380, +0.08798742],
		[+0.01558709, +2.05285700],
		[-1.05991530, +0.47591877],
		[-0.38967478, -1.50155930],
		[+0.28056574, +1.19688750],
		[+0.07793474, -0.64558935],
		[+1.18461110, -1.27384350],
		[-0.76376295, -0.39265870]]
	*/

	EXPECT_TRUE(abs(in_err.GetReal(0, 0) - +0.65465380) < 0.001);
	EXPECT_TRUE(abs(in_err.GetReal(1, 0) - +0.01558709) < 0.001);
	EXPECT_TRUE(abs(in_err.GetReal(2, 0) - -1.05991530) < 0.001);
	EXPECT_TRUE(abs(in_err.GetReal(3, 0) - -0.38967478) < 0.001);
	EXPECT_TRUE(abs(in_err.GetReal(4, 0) - +0.28056574) < 0.001);
	EXPECT_TRUE(abs(in_err.GetReal(5, 0) - +0.07793474) < 0.001);
	EXPECT_TRUE(abs(in_err.GetReal(6, 0) - +1.18461110) < 0.001);
	EXPECT_TRUE(abs(in_err.GetReal(7, 0) - -0.76376295) < 0.001);

	EXPECT_TRUE(abs(in_err.GetReal(0, 1) - +0.08798742) < 0.001);
	EXPECT_TRUE(abs(in_err.GetReal(1, 1) - +2.05285700) < 0.001);
	EXPECT_TRUE(abs(in_err.GetReal(2, 1) - +0.47591877) < 0.001);
	EXPECT_TRUE(abs(in_err.GetReal(3, 1) - -1.50155930) < 0.001);
	EXPECT_TRUE(abs(in_err.GetReal(4, 1) - +1.19688750) < 0.001);
	EXPECT_TRUE(abs(in_err.GetReal(5, 1) - -0.64558935) < 0.001);
	EXPECT_TRUE(abs(in_err.GetReal(6, 1) - -1.27384350) < 0.001);
	EXPECT_TRUE(abs(in_err.GetReal(7, 1) - -0.39265870) < 0.001);

	for (int i = 0; i < 8; i++) {
//		std::cout << exp_norm0.dx[i] << std::endl;
		EXPECT_TRUE(abs(in_err.GetReal(i, 0) - exp_norm0.dx[i]) < 0.000001);
		EXPECT_TRUE(abs(in_err.GetReal(i, 1) - exp_norm1.dx[i]) < 0.000001);
	}

#endif

}



TEST(NeuralNetBatchNormalizationAvxTest, testBatchNormalizationCmp)
{
	const int node_size = 7;
	const int frame_size = 1025;

	std::vector< SimpleBatchNorm<> > exp_norm(node_size, SimpleBatchNorm<>(frame_size));
	

	bb::NeuralNetBatchNormalizationAvx<> batch_norm0(node_size);
	bb::NeuralNetBatchNormalization<>    batch_norm1(node_size);
	batch_norm0.SetBatchSize(frame_size);
	batch_norm1.SetBatchSize(frame_size);
	testSetupLayerBuffer(batch_norm0);
	testSetupLayerBuffer(batch_norm1);

	auto in_sig0 = batch_norm0.GetInputSignalBuffer();
	auto in_sig1 = batch_norm1.GetInputSignalBuffer();
	int index = 11;
	for (int node = 0; node < node_size; ++node) {
		for (int frame = 0; frame < frame_size; ++frame) {
			in_sig0.SetReal(frame, node, (float)index);
			in_sig1.SetReal(frame, node, (float)index);
			exp_norm[node].x[frame] = (float)index;
			index++;
		}
	}

	batch_norm0.Forward(true);
	batch_norm1.Forward(true);
	for (int node = 0; node < node_size; ++node) {
		exp_norm[node].Forward();
	}

	auto out_sig0 = batch_norm0.GetOutputSignalBuffer();
	auto out_sig1 = batch_norm1.GetOutputSignalBuffer();

#if 0
	std::cout << out_sig0.GetReal(0, 0) << std::endl;
	std::cout << out_sig0.GetReal(1, 0) << std::endl;
	std::cout << out_sig0.GetReal(2, 0) << std::endl;
	std::cout << out_sig0.GetReal(3, 0) << std::endl;
	std::cout << out_sig0.GetReal(4, 0) << std::endl;
	std::cout << out_sig0.GetReal(5, 0) << std::endl;
	std::cout << out_sig0.GetReal(6, 0) << std::endl;
	std::cout << out_sig0.GetReal(7, 0) << std::endl;

	std::cout << out_sig1.GetReal(0, 0) << std::endl;
	std::cout << out_sig1.GetReal(1, 0) << std::endl;
	std::cout << out_sig1.GetReal(2, 0) << std::endl;
	std::cout << out_sig1.GetReal(3, 0) << std::endl;
	std::cout << out_sig1.GetReal(4, 0) << std::endl;
	std::cout << out_sig1.GetReal(5, 0) << std::endl;
	std::cout << out_sig1.GetReal(6, 0) << std::endl;
	std::cout << out_sig1.GetReal(7, 0) << std::endl;
#endif


	for (int node = 0; node < node_size; ++node) {
		for (int frame = 0; frame < frame_size; ++frame) {
			EXPECT_TRUE(abs(out_sig0.GetReal(frame, node) - out_sig1.GetReal(frame, node)) < 0.0001);
			EXPECT_TRUE(abs(out_sig0.GetReal(frame, node) - exp_norm[node].y[frame]) < 0.001);
		}
	}


	//	batch_norm.Forward(false);

}


