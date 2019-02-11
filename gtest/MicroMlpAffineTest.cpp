#include <string>
#include <iostream>
#include <fstream>

#include "gtest/gtest.h"

#include "bb/MicroMlpAffine.h"


TEST(MicroMlpAffineTest, testMicroMlpAffine)
{
	auto mlp = bb::MicroMlpAffine<4, 2>::Create(16);

    bb::FrameBuffer x(BB_TYPE_FP32, 32, 16);

    mlp->SetInputShape({16});
    auto y = mlp->Forward(x);
}


#if 0
inline void testSetupLayerBuffer(bb::NeuralNetLayer<>& net)
{
	net.SetInputSignalBuffer (net.CreateInputSignalBuffer());
	net.SetInputErrorBuffer (net.CreateInputErrorBuffer());
	net.SetOutputSignalBuffer(net.CreateOutputSignalBuffer());
	net.SetOutputErrorBuffer(net.CreateOutputErrorBuffer());
}


template <typename T = float>
class ModelAffine
{
public:
	std::vector<T>	W;
	T				b;
	std::vector<T>	dW;
	T				db;

	int				m_n;
	std::vector<T>	m_in_sig;

	explicit ModelAffine(int n) : m_n(n), W(n), dW(n, 0), db(0) {}

	T Forward(std::vector<T> in_sig)
	{
		m_in_sig = in_sig;

		T out_sig = b;
		for (int i = 0; i < m_n; ++i) {
			out_sig += W[i] * in_sig[i];
		}
		return out_sig;
	}

	std::vector<T> Backward(T out_err)
	{
		std::vector<T> in_err(m_n);
		db = out_err;
		for (int i = 0; i < m_n; ++i) {
			in_err[i] = out_err * W[i];
			dW[i]     = out_err * m_in_sig[i];
		}
		return in_err;
	}
};

template <typename T = float>
class ModelReLU
{
public:
	int				m_n;
	std::vector<T>	m_in_sig;

	explicit ModelReLU(int n) : m_n(n) {}

	std::vector<T> Foward(std::vector<T> in_sig)
	{
		m_in_sig = in_sig;

		std::vector<T> out_sig(m_n);
		for (int i = 0; i < m_n; ++i) {
			out_sig[i] = std::max(in_sig[i], (T)0);
		}
		return out_sig;
	}

	std::vector<T> Backward(std::vector<T> out_err)
	{
		std::vector<T> in_err(m_n);
		for (int i = 0; i < m_n; ++i) {
			in_err[i] = m_in_sig[i] > 0 ? out_err[i] : 0;
		}
		return in_err;
	}
};


TEST(NeuralNetStackedMicroAffineTest, testNeuralNetStackedMicroAffine)
{
	bb::NeuralNetStackedMicroAffine<4, 2> lut(6, 1);
	lut.SetBatchSize(1);
	testSetupLayerBuffer(lut);
	
	for (int i = 0; i < 1; i++) {
		for (int j = 0; j < 4; j++) {
			lut.SetNodeInput(i, j, j);
		}
	}

	auto in_sig  = lut.GetInputSignalBuffer();
	auto out_sig = lut.GetOutputSignalBuffer();
	auto in_err  = lut.GetInputErrorBuffer();
	auto out_err = lut.GetOutputErrorBuffer();
	
	ModelAffine<float>	affine0_0(4);
	ModelAffine<float>	affine0_1(4);
	ModelReLU<float>	relu(2);
	ModelAffine<float>	affine1(2);

	std::vector<float>	in_sig_val(4);
	float				out_err_val;
	
	in_sig_val[0] = 1;
	in_sig_val[1] = 2;
	in_sig_val[2] = 3;
	in_sig_val[3] = 4;
	out_err_val = 2;
	
	in_sig.SetReal(0, 0, in_sig_val[0]);
	in_sig.SetReal(0, 1, in_sig_val[1]);
	in_sig.SetReal(0, 2, in_sig_val[2]);
	in_sig.SetReal(0, 3, in_sig_val[3]);

	float W0[2][4];
	float b0[2];
	float W1[2];
	float b1;

	W0[0][0] = 2;
	W0[0][1] = 3;
	W0[0][2] = 4;
	W0[0][3] = 5;
	b0[0] = 1;

	W0[1][0] = 6;
	W0[1][1] = -7;
	W0[1][2] = -8;
	W0[1][3] = -9;
	b0[1] = 2;

	W1[0] = 5;
	W1[1] = 6;
	b1 = 1;

	for (int i = 0; i < 4; i++) {
		lut.W0(0, 0, i)   = W0[0][i];
		affine0_0.W[i] = W0[0][i];
	}
	lut.b0(0, 0) = b0[0];
	affine0_0.b = b0[0];

	for (int i = 0; i < 4; i++) {
		lut.W0(0, 1, i)   = W0[1][i];
		affine0_1.W[i] = W0[1][i];
	}
	lut.b0(0, 1) = b0[1];
	affine0_1.b = b0[1];

	for (int i = 0; i < 2; i++) {
		lut.W1(0, i) = W1[i];
		affine1.W[i] = W1[i];
	}
	lut.b1(0) = b1;
	affine1.b = b1;


	lut.Forward();

	std::vector<float> hidden_sig0(2);
	hidden_sig0[0] = affine0_0.Forward(in_sig_val);
	hidden_sig0[1] = affine0_1.Forward(in_sig_val);
	auto hidden_sig1 = relu.Foward(hidden_sig0);
	float out_sig_val = affine1.Forward(hidden_sig1);

	EXPECT_EQ(out_sig_val, out_sig.GetReal(0, 0));
	
	out_err.SetReal(0, 0, out_err_val);
	lut.Backward();

	auto hidden_err1 = affine1.Backward(out_err_val);
	auto hidden_err0 = relu.Backward(hidden_err1);
	auto in_err_val0 = affine0_0.Backward(hidden_err0[0]);
	auto in_err_val1 = affine0_0.Backward(hidden_err0[1]);

	EXPECT_EQ(in_err_val0[0] + in_err_val1[0], in_err.GetReal(0, 0));
	EXPECT_EQ(in_err_val0[1] + in_err_val1[1], in_err.GetReal(0, 1));
	EXPECT_EQ(in_err_val0[2] + in_err_val1[2], in_err.GetReal(0, 2));
	EXPECT_EQ(in_err_val0[3] + in_err_val1[3], in_err.GetReal(0, 3));


	lut.Update();
}

#endif
