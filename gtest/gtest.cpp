#include <stdio.h>
#include <iostream>
#include "gtest/gtest.h"
#include "bb/NeuralNetAffine.h"
#include "bb/NeuralNetSigmoid.h"
#include "bb/NeuralNetSoftmax.h"
#include "bb/NeuralNetBinarize.h"
#include "bb/NeuralNetUnbinarize.h"
#include "bb/NeuralNetBinaryLut6.h"
#include "bb/NeuralNetBinaryLutN.h"
#include "bb/NeuralNetConvolution.h"



inline void testSetupLayerBuffer(bb::NeuralNetLayer<>& net)
{
	net.SetInputValueBuffer (net.CreateInputValueBuffer());
	net.SetInputErrorBuffer (net.CreateInputErrorBuffer());
	net.SetOutputValueBuffer(net.CreateOutputValueBuffer());
	net.SetOutputErrorBuffer(net.CreateOutputErrorBuffer());
}


TEST(NeuralNetAffineTest, testAffine)
{
	bb::NeuralNetAffine<> affine(2, 3);
	testSetupLayerBuffer(affine);

//	float in[2] = { 1, 2 };
//	float out[3];
//	affine.SetInputValuePtr(in);
//	affine.SetOutputValuePtr(out);

	auto in_val = affine.GetInputValueBuffer();
	auto out_val = affine.GetOutputValueBuffer();

	in_val.SetReal(0, 0, 1);
	in_val.SetReal(0, 1, 2);
	EXPECT_EQ(1, in_val.GetReal(0, 0));
	EXPECT_EQ(2, in_val.GetReal(0, 1));

	affine.W(0, 0) = 1;
	affine.W(1, 0) = 2;
	affine.W(0, 1) = 10;
	affine.W(1, 1) = 20;
	affine.W(0, 2) = 100;
	affine.W(1, 2) = 200;
	affine.b(0) = 1000;
	affine.b(1) = 2000;
	affine.b(2) = 3000;
	affine.Forward();
	EXPECT_EQ(1 * 1 + 2 * 2 + 1000, out_val.GetReal(0, 0));
	EXPECT_EQ(1 * 10 + 2 * 20 + 2000, out_val.GetReal(0, 1));
	EXPECT_EQ(1 * 100 + 2 * 200 + 3000, out_val.GetReal(0, 2));

//	float outError[3] = { 998, 2042, 3491 };
//	float inError[2];
//	affine.SetOutputErrorPtr(outError);
//	affine.SetInputErrorPtr(inError);

	auto in_err = affine.GetInputErrorBuffer();
	auto out_err = affine.GetOutputErrorBuffer();
	out_err.SetReal(0, 0, 998);
	out_err.SetReal(0, 1, 2042);
	out_err.SetReal(0, 2, 3491);

	affine.Backward();
	EXPECT_EQ(370518, in_err.GetReal(0, 0));
	EXPECT_EQ(741036, in_err.GetReal(0, 1));

	EXPECT_EQ(998,  affine.dW(0, 0));
	EXPECT_EQ(2042, affine.dW(0, 1));
	EXPECT_EQ(3491, affine.dW(0, 2));
	EXPECT_EQ(1996, affine.dW(1, 0));
	EXPECT_EQ(4084, affine.dW(1, 1));
	EXPECT_EQ(6982, affine.dW(1, 2));

//	std::cout << "W = \n" << affine.m_W << std::endl;
//	std::cout << "dW = \n" << affine.m_dW << std::endl;
//	std::cout << "b = \n" << affine.m_b << std::endl;
//	std::cout << "db = \n" << affine.m_db << std::endl;
	
	affine.Update(0.1);
}


TEST(NeuralNetAffineTest, testAffineBatch)
{
	bb::NeuralNetAffine<> affine(2, 3, 2);
	testSetupLayerBuffer(affine);

//	float in[2*2] = { 1, 3, 2, 4 };
//	float out[3*2];
//	affine.SetInputValuePtr(in);
//	affine.SetOutputValuePtr(out);

	auto in_val = affine.GetInputValueBuffer();
	auto out_val = affine.GetOutputValueBuffer();
	in_val.SetReal(0, 0, 1);
	in_val.SetReal(0, 1, 2);
	in_val.SetReal(1, 0, 3);
	in_val.SetReal(1, 1, 4);


	affine.W(0, 0) = 1;
	affine.W(1, 0) = 2;
	affine.W(0, 1) = 10;
	affine.W(1, 1) = 20;
	affine.W(0, 2) = 100;
	affine.W(1, 2) = 200;
	affine.b(0) = 1000;
	affine.b(1) = 2000;
	affine.b(2) = 3000;
	affine.Forward();
		
	EXPECT_EQ(1 * 1 + 2 * 2 + 1000, out_val.GetReal(0, 0));
	EXPECT_EQ(1 * 10 + 2 * 20 + 2000, out_val.GetReal(0, 1));
	EXPECT_EQ(1 * 100 + 2 * 200 + 3000, out_val.GetReal(0, 2));
	EXPECT_EQ(3 * 1 + 4 * 2 + 1000, out_val.GetReal(1, 0));
	EXPECT_EQ(3 * 10 + 4 * 20 + 2000, out_val.GetReal(1, 1));
	EXPECT_EQ(3 * 100 + 4 * 200 + 3000, out_val.GetReal(1, 2));
	
//	float outError[3 * 2] = { 998, 1004, 2042, 2102, 3491, 4091 };
//	float inError[2 * 2];
//	affine.SetOutputErrorPtr(outError);
//	affine.SetInputErrorPtr(inError);

	auto out_err = affine.GetOutputErrorBuffer();
	auto in_err = affine.GetInputErrorBuffer();
	out_err.SetReal(0, 0, 998);
	out_err.SetReal(1, 0, 1004);
	out_err.SetReal(0, 1, 2042);
	out_err.SetReal(1, 1, 2102);
	out_err.SetReal(0, 2, 3491);
	out_err.SetReal(1, 2, 4091);

	affine.Backward();
//	EXPECT_EQ(370518, inError[0]);
//	EXPECT_EQ(741036, inError[2]);
//	EXPECT_EQ(431124, inError[1]);
//	EXPECT_EQ(862248, inError[3]);

	EXPECT_EQ(370518, in_err.GetReal(0, 0));
	EXPECT_EQ(741036, in_err.GetReal(0, 1));
	EXPECT_EQ(431124, in_err.GetReal(1, 0));
	EXPECT_EQ(862248, in_err.GetReal(1, 1));
}



TEST(NeuralNetSigmoidTest, testSigmoid)
{
	bb::NeuralNetSigmoid<> sigmoid(2);
	testSetupLayerBuffer(sigmoid);

//	float in[2] = { 1, 2};
//	float out[2];
//	sigmoid.SetInputValuePtr(in);
//	sigmoid.SetOutputValuePtr(out);

	auto in_val = sigmoid.GetInputValueBuffer();
	auto out_val = sigmoid.GetOutputValueBuffer();
	in_val.SetReal(0, 0, 1);
	in_val.SetReal(0, 1, 2);

	sigmoid.Forward();
	EXPECT_EQ(1.0f / (1.0f + exp(-1.0f)), out_val.GetReal(0, 0));
	EXPECT_EQ(1.0f / (1.0f + exp(-2.0f)), out_val.GetReal(0, 1));

//	float outError[2] = { 2, 3 };
//	float inError[2];
//	sigmoid.SetOutputErrorPtr(outError);
//	sigmoid.SetInputErrorPtr(inError);

	auto out_err = sigmoid.GetOutputErrorBuffer();
	auto in_err = sigmoid.GetInputErrorBuffer();
	out_err.SetReal(0, 0, 2);
	out_err.SetReal(0, 1, 3);


	sigmoid.Backward();

//	EXPECT_EQ(outError[0] * (1.0f - out[0]) * out[0], inError[0]);
//	EXPECT_EQ(outError[1] * (1.0f - out[1]) * out[1], inError[1]);

	EXPECT_EQ(out_err.GetReal(0, 0) * (1.0f - out_val.GetReal(0, 0)) * out_val.GetReal(0, 0), in_err.GetReal(0, 0));
	EXPECT_EQ(out_err.GetReal(0, 1) * (1.0f - out_val.GetReal(0, 1)) * out_val.GetReal(0, 1), in_err.GetReal(0, 1));
}


TEST(NeuralNetSigmoidTest, testSigmoidBatch)
{
	bb::NeuralNetSigmoid<> sigmoid(2, 2);
	testSetupLayerBuffer(sigmoid);

//	float in[2*2] = { 1, 2, 3, 4 };
//	float out[2 * 2];
//	sigmoid.SetInputValuePtr(in);
//	sigmoid.SetOutputValuePtr(out);

	auto in_val = sigmoid.GetInputValueBuffer();
	auto out_val = sigmoid.GetOutputValueBuffer();
	in_val.SetReal(0, 0, 1);
	in_val.SetReal(1, 0, 2);
	in_val.SetReal(0, 1, 3);
	in_val.SetReal(1, 1, 4);

	sigmoid.Forward();
	EXPECT_EQ(1.0f / (1.0f + exp(-1.0f)), out_val.GetReal(0, 0));
	EXPECT_EQ(1.0f / (1.0f + exp(-2.0f)), out_val.GetReal(1, 0));
	EXPECT_EQ(1.0f / (1.0f + exp(-3.0f)), out_val.GetReal(0, 1));
	EXPECT_EQ(1.0f / (1.0f + exp(-4.0f)), out_val.GetReal(1, 1));


//	float outError[2*2] = { 2, 3, 4, -5 };
//	float inError[2*2];
//	sigmoid.SetOutputErrorPtr(outError);
//	sigmoid.SetInputErrorPtr(inError);

	auto out_err = sigmoid.GetOutputErrorBuffer();
	auto in_err = sigmoid.GetInputErrorBuffer();
	out_err.SetReal(0, 0, 2);
	out_err.SetReal(1, 0, 3);
	out_err.SetReal(0, 1, 4);
	out_err.SetReal(1, 1, -5);


	sigmoid.Backward();

	EXPECT_EQ(out_err.GetReal(0, 0) * (1.0f - out_val.GetReal(0, 0)) * out_val.GetReal(0, 0), in_err.GetReal(0, 0));
	EXPECT_EQ(out_err.GetReal(1, 0) * (1.0f - out_val.GetReal(1, 0)) * out_val.GetReal(1, 0), in_err.GetReal(1, 0));
	EXPECT_EQ(out_err.GetReal(0, 1) * (1.0f - out_val.GetReal(0, 1)) * out_val.GetReal(0, 1), in_err.GetReal(0, 1));
	EXPECT_EQ(out_err.GetReal(1, 1) * (1.0f - out_val.GetReal(1, 1)) * out_val.GetReal(1, 1), in_err.GetReal(1, 1));
}


class NeuralNetSoftmaxTest : public ::testing::Test {
protected:
	Eigen::MatrixXf				m_src;
	Eigen::Matrix<float, -1, 1> m_max;
	Eigen::MatrixXf				m_norm;
	Eigen::MatrixXf				m_exp;
	Eigen::Matrix<float, -1, 1> m_sum;
	Eigen::MatrixXf				m_softmax;

	virtual void SetUp() {
		m_src = Eigen::MatrixXf(2, 3);
		m_src(0, 0) = 1010.0f;
		m_src(0, 1) = 1000.0f;
		m_src(0, 2) = 990.0f;
		m_src(1, 0) = 0.2f;
		m_src(1, 1) = 0.5f;
		m_src(1, 2) = 0.1f;
		m_max = m_src.rowwise().maxCoeff();
		m_norm = m_src.colwise() - m_max;
		m_exp = m_norm.array().exp();
		m_sum = m_exp.rowwise().sum();
		m_softmax = m_exp.array().colwise() / m_sum.array();

#if 0
		std::cout << "m_src = \n" << m_src << std::endl;
		std::cout << "m_max = \n" << m_max << std::endl;
		std::cout << "m_norm = \n" << m_norm << std::endl;
		std::cout << "m_exp = \n" << m_exp << std::endl;
		std::cout << "m_sum = \n" << m_sum << std::endl;
		std::cout << "m_softmax = \n" << m_softmax << std::endl;
#endif
	}
};



TEST_F(NeuralNetSoftmaxTest, testSoftmax)
{
	bb::NeuralNetSoftmax<> softmax(3);
	testSetupLayerBuffer(softmax);

//	float in[3];
//	float out[3];
//	in[0] = m_src(0, 0);
//	in[1] = m_src(0, 1);
//	in[2] = m_src(0, 2);
//	softmax.SetInputValuePtr(in);
//	softmax.SetOutputValuePtr(out);

	auto in_val = softmax.GetInputValueBuffer();
	auto out_val = softmax.GetOutputValueBuffer();
	in_val.SetReal(0, 0, m_src(0, 0));
	in_val.SetReal(0, 1, m_src(0, 1));
	in_val.SetReal(0, 2, m_src(0, 2));

	softmax.Forward();

//	EXPECT_EQ(m_softmax(0, 0), out[0]);
//	EXPECT_EQ(m_softmax(0, 1), out[1]);
//	EXPECT_EQ(m_softmax(0, 2), out[2]);
	EXPECT_EQ(m_softmax(0, 0), out_val.GetReal(0, 0));
	EXPECT_EQ(m_softmax(0, 1), out_val.GetReal(0, 1));
	EXPECT_EQ(m_softmax(0, 2), out_val.GetReal(0, 2));

//	float outError[3] = { 2, 3, 4 };
//	float inError[3];
//	softmax.SetOutputErrorPtr(outError);
//	softmax.SetInputErrorPtr(inError);

	auto out_err = softmax.GetOutputErrorBuffer();
	auto in_err = softmax.GetInputErrorBuffer();
	out_err.SetReal(0, 0, 2);
	out_err.SetReal(0, 1, 3);
	out_err.SetReal(0, 2, 4);

	softmax.Backward();

//	EXPECT_EQ(2, inError[0]);
//	EXPECT_EQ(3, inError[1]);
//	EXPECT_EQ(4, inError[2]);
	EXPECT_EQ(2, in_err.GetReal(0, 0));
	EXPECT_EQ(3, in_err.GetReal(0, 1));
	EXPECT_EQ(4, in_err.GetReal(0, 2));
}


TEST_F(NeuralNetSoftmaxTest, testSoftmaxBatch)
{
	bb::NeuralNetSoftmax<> softmax(3, 2);
	testSetupLayerBuffer(softmax);

//	float in[3*2];
//	float out[3*2];
//	in[0] = m_src(0, 0);
//	in[1] = m_src(1, 0);
//	in[2] = m_src(0, 1);
//	in[3] = m_src(1, 1);
//	in[4] = m_src(0, 2);
//	in[5] = m_src(1, 2);
//	softmax.SetInputValuePtr(in);
//	softmax.SetOutputValuePtr(out);

	auto in_val = softmax.GetInputValueBuffer();
	auto out_val = softmax.GetOutputValueBuffer();
	in_val.SetReal(0, 0, m_src(0, 0));
	in_val.SetReal(1, 0, m_src(1, 0));
	in_val.SetReal(0, 1, m_src(0, 1));
	in_val.SetReal(1, 1, m_src(1, 1));
	in_val.SetReal(0, 2, m_src(0, 2));
	in_val.SetReal(1, 2, m_src(1, 2));

	softmax.Forward();

//	EXPECT_EQ(m_softmax(0, 0), out[0]);
//	EXPECT_EQ(m_softmax(1, 0), out[1]);
//	EXPECT_EQ(m_softmax(0, 1), out[2]);
//	EXPECT_EQ(m_softmax(1, 1), out[3]);
//	EXPECT_EQ(m_softmax(0, 2), out[4]);
//	EXPECT_EQ(m_softmax(1, 2), out[5]);

	EXPECT_EQ(m_softmax(0, 0), out_val.GetReal(0, 0));
	EXPECT_EQ(m_softmax(1, 0), out_val.GetReal(1, 0));
	EXPECT_EQ(m_softmax(0, 1), out_val.GetReal(0, 1));
	EXPECT_EQ(m_softmax(1, 1), out_val.GetReal(1, 1));
	EXPECT_EQ(m_softmax(0, 2), out_val.GetReal(0, 2));
	EXPECT_EQ(m_softmax(1, 2), out_val.GetReal(1, 2));


//	float outError[3*2] = { 2, 3, 4, 5, 6, 7 };
//	float inError[3 * 2];
//	softmax.SetOutputErrorPtr(outError);
//	softmax.SetInputErrorPtr(inError);

	auto out_err = softmax.GetOutputErrorBuffer();
	auto in_err = softmax.GetInputErrorBuffer();
	out_err.SetReal(0, 0, 2);
	out_err.SetReal(1, 0, 3);
	out_err.SetReal(0, 1, 4);
	out_err.SetReal(1, 1, 5);
	out_err.SetReal(0, 2, 6);
	out_err.SetReal(1, 2, 7);

	softmax.Backward();

//	EXPECT_EQ(2, inError[0]);
//	EXPECT_EQ(3, inError[1]);
//	EXPECT_EQ(4, inError[2]);
//	EXPECT_EQ(5, inError[3]);
//	EXPECT_EQ(6, inError[4]);
//	EXPECT_EQ(7, inError[5]);

	EXPECT_EQ(2, in_err.GetReal(0, 0));
	EXPECT_EQ(3, in_err.GetReal(1, 0));
	EXPECT_EQ(4, in_err.GetReal(0, 1));
	EXPECT_EQ(5, in_err.GetReal(1, 1));
	EXPECT_EQ(6, in_err.GetReal(0, 2));
	EXPECT_EQ(7, in_err.GetReal(1, 2));
}


TEST(NeuralNetBinarizeTest, testNeuralNetBinarize)
{
	const int node_size = 3;
	const int mux_size = 2;
	const int frame_size = 1;

	bb::NeuralNetBinarize<> binarize(node_size, node_size, mux_size);
	testSetupLayerBuffer(binarize);

	EXPECT_EQ(1, binarize.GetInputFrameSize());
	EXPECT_EQ(2, binarize.GetOutputFrameSize());


//	float	in[node_size];
//	__m256i out[(((frame_size * mux_size) + 255) / 256) * node_size];
//	NeuralNetBufferAccessorReal<> accReal(in, 1);
//	NeuralNetBufferAccessorBinary<> accBin(out, 2);
//	accReal.Set(0, 0, 0.0f);
//	accReal.Set(0, 1, 1.0f);
//	accReal.Set(0, 2, 0.5f);
//	binarize.SetInputValuePtr(in);
//	binarize.SetOutputValuePtr(out);

	auto in_val = binarize.GetInputValueBuffer();
	auto out_val = binarize.GetOutputValueBuffer();
	in_val.SetReal(0, 0, 0.0f);
	in_val.SetReal(0, 1, 1.0f);
	in_val.SetReal(0, 2, 0.5f);

	binarize.Forward();

//	EXPECT_EQ(false, accBin.Get(0, 0));
//	EXPECT_EQ(false, accBin.Get(1, 0));
//	EXPECT_EQ(true, accBin.Get(0, 1));
//	EXPECT_EQ(true, accBin.Get(1, 1));

	EXPECT_EQ(false, out_val.GetBinary(0, 0));
	EXPECT_EQ(false, out_val.GetBinary(1, 0));
	EXPECT_EQ(true, out_val.GetBinary(0, 1));
	EXPECT_EQ(true, out_val.GetBinary(1, 1));

//	__m256i outError[(((frame_size * mux_size) + 255) / 256) * node_size];
//	float	inError[node_size];
//	NeuralNetBufferAccessorReal<> accRealErr(inError, 1);
//	NeuralNetBufferAccessorBinary<> accBinErr(outError, 2);
//	accBinErr.Set(0, 0, false);
//	accBinErr.Set(1, 0, false);
//	accBinErr.Set(0, 1, true);
//	accBinErr.Set(1, 1, true);
//	accBinErr.Set(0, 2, true);
//	accBinErr.Set(1, 2, false);
//	binarize.SetOutputErrorPtr(outError);
//	binarize.SetInputErrorPtr(inError);

	auto out_err = binarize.GetOutputErrorBuffer();
	auto in_err = binarize.GetInputErrorBuffer();

	out_err.SetBinary(0, 0, false);
	out_err.SetBinary(1, 0, false);
	out_err.SetBinary(0, 1, true);
	out_err.SetBinary(1, 1, true);
	out_err.SetBinary(0, 2, true);
	out_err.SetBinary(1, 2, false);


	binarize.Backward();
	
//	EXPECT_EQ(0.0, accRealErr.Get(0, 0));
//	EXPECT_EQ(1.0, accRealErr.Get(0, 1));
//	EXPECT_EQ(0.5, accRealErr.Get(0, 2));

	EXPECT_EQ(0.0f, in_err.GetReal(0, 0));
	EXPECT_EQ(1.0f, in_err.GetReal(0, 1));
	EXPECT_EQ(0.5f, in_err.Get<float>(0, 2));
}


TEST(NeuralNetBinarizeTest, testNeuralNetBinarizeBatch)
{
	const int node_size = 3;
	const int mux_size = 2;
	const int batch_size = 2;

	bb::NeuralNetBinarize<> binarize(node_size, node_size, mux_size, batch_size);
	testSetupLayerBuffer(binarize);

	EXPECT_EQ(batch_size, binarize.GetInputFrameSize());
	EXPECT_EQ(batch_size*mux_size, binarize.GetOutputFrameSize());

//	float	in[batch_size*node_size];
//	__m256i out[(((batch_size * mux_size) + 255) / 256) * node_size];
//	NeuralNetBufferAccessorReal<> accReal(in, batch_size);
//	NeuralNetBufferAccessorBinary<> accBin(out, batch_size*mux_size);
//	accReal.Set(0, 0, 0.0f);
//	accReal.Set(0, 1, 1.0f);
//	accReal.Set(0, 2, 0.5f);
//	accReal.Set(1, 0, 1.0f);
//	accReal.Set(1, 1, 0.5f);
//	accReal.Set(1, 2, 0.0f);
//	binarize.SetInputValuePtr(in);
//	binarize.SetOutputValuePtr(out);

	auto in_val = binarize.GetInputValueBuffer();
	auto out_val = binarize.GetOutputValueBuffer();
	in_val.SetReal(0, 0, 0.0f);
	in_val.SetReal(0, 1, 1.0f);
	in_val.SetReal(0, 2, 0.5f);
	in_val.SetReal(1, 0, 1.0f);
	in_val.SetReal(1, 1, 0.5f);
	in_val.SetReal(1, 2, 0.0f);
	
	binarize.Forward();
	
//	EXPECT_EQ(true, accBin.Get(2, 0));
//	EXPECT_EQ(true, accBin.Get(3, 0));
//	EXPECT_EQ(false, accBin.Get(2, 2));
//	EXPECT_EQ(false, accBin.Get(3, 2));

	EXPECT_EQ(true, out_val.GetBinary(2, 0));
	EXPECT_EQ(true, out_val.Get<bool>(3, 0));
	EXPECT_EQ(false, out_val.GetBinary(2, 2));
	EXPECT_EQ(false, out_val.Get<bool>(3, 2));


//	__m256i outError[(((batch_size * mux_size) + 255) / 256) * node_size];
//	float	inError[batch_size*node_size];
//	NeuralNetBufferAccessorReal<> accRealErr(inError, batch_size);
//	NeuralNetBufferAccessorBinary<> accBinErr(outError, batch_size*mux_size);
//
//	accBinErr.Set(0, 0, false);
//	accBinErr.Set(1, 0, false);
//	accBinErr.Set(0, 1, true);
//	accBinErr.Set(1, 1, true);
//	accBinErr.Set(0, 2, true);
//	accBinErr.Set(1, 2, false);
//
//	accBinErr.Set(2, 0, true);
//	accBinErr.Set(3, 0, false);
//	accBinErr.Set(2, 1, false);
//	accBinErr.Set(3, 1, false);
//	accBinErr.Set(2, 2, true);
//	accBinErr.Set(3, 2, true);
//	binarize.SetOutputErrorPtr(outError);
//	binarize.SetInputErrorPtr(inError);

	auto out_err = binarize.GetOutputErrorBuffer();
	auto in_err = binarize.GetInputErrorBuffer();

	out_err.SetBinary(0, 0, false);
	out_err.SetBinary(1, 0, false);
	out_err.SetBinary(0, 1, true);
	out_err.SetBinary(1, 1, true);
	out_err.SetBinary(0, 2, true);
	out_err.SetBinary(1, 2, false);

	out_err.SetBinary(2, 0, true);
	out_err.SetBinary(3, 0, false);
	out_err.SetBinary(2, 1, false);
	out_err.SetBinary(3, 1, false);
	out_err.SetBinary(2, 2, true);
	out_err.SetBinary(3, 2, true);
	
	binarize.Backward();
	
//	EXPECT_EQ(0.0, accRealErr.Get(0, 0));
//	EXPECT_EQ(1.0, accRealErr.Get(0, 1));
//	EXPECT_EQ(0.5, accRealErr.Get(0, 2));

//	EXPECT_EQ(0.5, accRealErr.Get(1, 0));
//	EXPECT_EQ(0.0, accRealErr.Get(1, 1));
//	EXPECT_EQ(1.0, accRealErr.Get(1, 2));
	
	EXPECT_EQ(0.0f, in_err.GetReal(0, 0));
	EXPECT_EQ(1.0f, in_err.GetReal(0, 1));
	EXPECT_EQ(0.5f, in_err.GetReal(0, 2));

	EXPECT_EQ(0.5f, in_err.GetReal(1, 0));
	EXPECT_EQ(0.0f, in_err.GetReal(1, 1));
	EXPECT_EQ(1.0f, in_err.GetReal(1, 2));
}


TEST(NeuralNetUnbinarizeTest, testNeuralNetUnbinarize)
{
	const int node_size = 3;
	const int mux_size = 2;
	const int frame_size = 1;

	bb::NeuralNetUnbinarize<> unbinarize(node_size, node_size, mux_size);
	testSetupLayerBuffer(unbinarize);

	EXPECT_EQ(2, unbinarize.GetInputFrameSize());
	EXPECT_EQ(1, unbinarize.GetOutputFrameSize());


//	__m256i in[(((frame_size * mux_size) + 255) / 256) * node_size];
//	float	out[node_size];
//	NeuralNetBufferAccessorBinary<> accBin(in, 2);
//	NeuralNetBufferAccessorReal<> accReal(out, 1);
//	accBin.Set(0, 0, true);
//	accBin.Set(1, 0, true);
//	accBin.Set(0, 1, false);
//	accBin.Set(1, 1, false);
//	accBin.Set(0, 2, false);
//	accBin.Set(1, 2, true);
//	unbinarize.SetInputValuePtr(in);
//	unbinarize.SetOutputValuePtr(out);

	auto in_val = unbinarize.GetInputValueBuffer();
	auto out_val = unbinarize.GetOutputValueBuffer();
	in_val.SetBinary(0, 0, true);
	in_val.SetBinary(1, 0, true);
	in_val.SetBinary(0, 1, false);
	in_val.SetBinary(1, 1, false);
	in_val.SetBinary(0, 2, false);
	in_val.SetBinary(1, 2, true);

	unbinarize.Forward();
	
//	EXPECT_EQ(1.0, accReal.Get(0, 0));
//	EXPECT_EQ(0.0, accReal.Get(0, 1));
//	EXPECT_EQ(0.5, accReal.Get(0, 2));

	EXPECT_EQ(1.0, out_val.GetReal(0, 0));
	EXPECT_EQ(0.0, out_val.GetReal(0, 1));
	EXPECT_EQ(0.5, out_val.GetReal(0, 2));
	
//	float	outError[node_size];
//	__m256i inError[(((frame_size * mux_size) + 255) / 256) * node_size];
//	NeuralNetBufferAccessorReal<> accRealErr(outError, 1);
//	NeuralNetBufferAccessorBinary<> accBinErr(inError, 2);
//	accRealErr.Set(0, 0, 0.0f);
//	accRealErr.Set(0, 1, 1.0f);
//	accRealErr.Set(0, 2, 0.5f);
//	unbinarize.SetOutputErrorPtr(outError);
//	unbinarize.SetInputErrorPtr(inError);

	auto out_err = unbinarize.GetOutputErrorBuffer();
	auto in_err = unbinarize.GetInputErrorBuffer();
	out_err.SetReal(0, 0, 0.0f);
	out_err.SetReal(0, 1, 1.0f);
	out_err.SetReal(0, 2, 0.5f);

	unbinarize.Backward();
//	EXPECT_EQ(false, accBinErr.Get(0, 0));
//	EXPECT_EQ(false, accBinErr.Get(1, 0));
//	EXPECT_EQ(true, accBinErr.Get(0, 1));
//	EXPECT_EQ(true, accBinErr.Get(1, 1));

	EXPECT_EQ(false, in_err.GetBinary(0, 0));
	EXPECT_EQ(false, in_err.GetBinary(1, 0));
	EXPECT_EQ(true,  in_err.GetBinary(0, 1));
	EXPECT_EQ(true,  in_err.GetBinary(1, 1));
}


#if 0
TEST(NeuralNetBinaryLut6, testNeuralNetBinaryLut6)
{
	bb::NeuralNetBinaryLut6<> lut(16, 2, 1, 1, 1);
	testSetupLayerBuffer(lut);

//	__m256i in[8];
//	__m256i out[2];
//	NeuralNetBufferAccessorBinary<> accIn(in, 1);
//	NeuralNetBufferAccessorBinary<> accOut(out, 1);
//	accIn.Set(0, 0, false);
//	accIn.Set(0, 1, true);
//	accIn.Set(0, 2, true);
//	accIn.Set(0, 3, false);
//	accIn.Set(0, 4, false);
//	accIn.Set(0, 5, true);
//	accIn.Set(0, 6, true);
//	accIn.Set(0, 7, true);

	auto in_val = lut.GetInputValueBuffer();
	auto out_val = lut.GetOutputValueBuffer();
	in_val.SetBinary(0, 0, false);
	in_val.SetBinary(0, 1, true);
	in_val.SetBinary(0, 2, true);
	in_val.SetBinary(0, 3, false);
	in_val.SetBinary(0, 4, false);
	in_val.SetBinary(0, 5, true);
	in_val.SetBinary(0, 6, true);
	in_val.SetBinary(0, 7, true);

	// 0x1d
	lut.SetLutInput(0, 0, 6);	// 1
	lut.SetLutInput(0, 1, 4);	// 0
	lut.SetLutInput(0, 2, 1);	// 1
	lut.SetLutInput(0, 3, 7);	// 1
	lut.SetLutInput(0, 4, 2);	// 1
	lut.SetLutInput(0, 5, 3);	// 0

	// 0x1c
	lut.SetLutInput(1, 0, 0);	// 0
	lut.SetLutInput(1, 1, 4);	// 0
	lut.SetLutInput(1, 2, 1);	// 1
	lut.SetLutInput(1, 3, 5);	// 1
	lut.SetLutInput(1, 4, 6);	// 1
	lut.SetLutInput(1, 5, 3);	// 0

	for (int i = 0; i < 64; i++) {
		lut.SetLutTable(0, i, i == 0x1d);
		lut.SetLutTable(0, i, i != 0x1c);
	}

//	lut.SetInputValuePtr(in);
//	lut.SetOutputValuePtr(out);
	
	lut.Forward();
//	EXPECT_EQ(true, accOut.Get(0, 0));
//	EXPECT_EQ(false, accOut.Get(0, 1));

	EXPECT_EQ(true, out_val.GetBinary(0, 0));
	EXPECT_EQ(false, out_val.Get<bool>(0, 1));
}



TEST(NeuralNetBinaryLut6, testNeuralNetBinaryLut6Batch)
{
	bb::NeuralNetBinaryLut6<> lut(16, 2, 2, 1, 1);
	testSetupLayerBuffer(lut);

//	__m256i in[8];
//	__m256i out[2];
//
//	NeuralNetBufferAccessorBinary<> accIn(in, 2);
//	NeuralNetBufferAccessorBinary<> accOut(out, 2);
//
//	accIn.Set(0, 0, false);
//	accIn.Set(0, 1, true);
//	accIn.Set(0, 2, true);
//	accIn.Set(0, 3, false);
//	accIn.Set(0, 4, false);
//	accIn.Set(0, 5, true);
//	accIn.Set(0, 6, true);
//	accIn.Set(0, 7, true);
//
//	accIn.Set(1, 0, true);
//	accIn.Set(1, 1, false);
//	accIn.Set(1, 2, false);
//	accIn.Set(1, 3, true);
//	accIn.Set(1, 4, true);
//	accIn.Set(1, 5, false);
//	accIn.Set(1, 6, false);
//	accIn.Set(1, 7, false);

	auto in_val = lut.GetInputValueBuffer();
	auto out_val = lut.GetOutputValueBuffer();
	in_val.SetBinary(0, 0, false);
	in_val.SetBinary(0, 1, true);
	in_val.SetBinary(0, 2, true);
	in_val.SetBinary(0, 3, false);
	in_val.SetBinary(0, 4, false);
	in_val.SetBinary(0, 5, true);
	in_val.SetBinary(0, 6, true);
	in_val.SetBinary(0, 7, true);

	in_val.SetBinary(1, 0, true);
	in_val.SetBinary(1, 1, false);
	in_val.SetBinary(1, 2, false);
	in_val.SetBinary(1, 3, true);
	in_val.SetBinary(1, 4, true);
	in_val.SetBinary(1, 5, false);
	in_val.SetBinary(1, 6, false);
	in_val.SetBinary(1, 7, false);

	// 0x1d
	lut.SetLutInput(0, 0, 6);	// 1
	lut.SetLutInput(0, 1, 4);	// 0
	lut.SetLutInput(0, 2, 1);	// 1
	lut.SetLutInput(0, 3, 7);	// 1
	lut.SetLutInput(0, 4, 2);	// 1
	lut.SetLutInput(0, 5, 3);	// 0

	// 0x1c
	lut.SetLutInput(1, 0, 0);	// 0
	lut.SetLutInput(1, 1, 4);	// 0
	lut.SetLutInput(1, 2, 1);	// 1
	lut.SetLutInput(1, 3, 5);	// 1
	lut.SetLutInput(1, 4, 6);	// 1
	lut.SetLutInput(1, 5, 3);	// 0

	for (int i = 0; i < 64; i++) {
		lut.SetLutTable(0, i, i == 0x1d || i == 0x22 );
		lut.SetLutTable(1, i, i != 0x1c && i != 0x23 );
	}

//	lut.SetInputValuePtr(in);
//	lut.SetOutputValuePtr(out);
	lut.Forward();
//	EXPECT_EQ(true, accOut.Get(0, 0));
//	EXPECT_EQ(false, accOut.Get(0, 1));
//	EXPECT_EQ(true, accOut.Get(1, 0));
//	EXPECT_EQ(false, accOut.Get(1, 1));

	EXPECT_EQ(true, out_val.GetBinary(0, 0));
	EXPECT_EQ(false, out_val.Get<bool>(0, 1));
	EXPECT_EQ(true, out_val.GetBinary(1, 0));
	EXPECT_EQ(false, out_val.Get<bool>(1, 1));
}


TEST(NeuralNetBinaryLut6, testNeuralNetBinaryLut6Compare)
{
	const size_t input_node_size  = 23;
	const size_t output_node_size = 77;
	const size_t mux_size = 23;
	const size_t batch_size = 345;
	const size_t frame_size = mux_size * batch_size;
	const int lut_input_size = 6;
	const int lut_table_size = 64;

	std::mt19937_64	mt(123);
	std::uniform_int<size_t>	rand_input(0, input_node_size - 1);
	std::uniform_int<int>		rand_bin(0, 1);

	bb::NeuralNetBinaryLut6<>  lut0(input_node_size, output_node_size, mux_size, batch_size, 1);
	bb::NeuralNetBinaryLutN<6> lut1(input_node_size, output_node_size, mux_size, batch_size, 1);

	testSetupLayerBuffer(lut0);
	testSetupLayerBuffer(lut1);

	EXPECT_EQ(input_node_size, lut0.GetInputNodeSize());
	EXPECT_EQ(output_node_size, lut0.GetOutputNodeSize());
	EXPECT_EQ(frame_size, lut0.GetInputFrameSize());
	EXPECT_EQ(frame_size, lut0.GetOutputFrameSize());
	EXPECT_EQ(lut_input_size, lut0.GetLutInputSize());
	EXPECT_EQ(lut_table_size, lut0.GetLutTableSize());
	EXPECT_EQ(lut0.GetInputNodeSize() , lut1.GetInputNodeSize());
	EXPECT_EQ(lut0.GetOutputNodeSize(), lut1.GetOutputNodeSize());
	EXPECT_EQ(lut0.GetInputFrameSize(), lut1.GetInputFrameSize());
	EXPECT_EQ(lut0.GetOutputFrameSize(), lut1.GetOutputFrameSize());
	EXPECT_EQ(lut0.GetLutInputSize(), lut1.GetLutInputSize());
	EXPECT_EQ(lut0.GetLutTableSize(), lut1.GetLutTableSize());

	// 設定
	for (size_t node = 0; node < output_node_size; ++node) {
		for (int lut_input = 0; lut_input < lut_input_size; ++lut_input) {
			size_t input_node = rand_input(mt);
			lut0.SetLutInput(node, lut_input, input_node);
			lut1.SetLutInput(node, lut_input, input_node);
		}

		for (int bit = 0; bit < lut_table_size; ++bit) {
			bool table_value = (rand_bin(mt) != 0);
			lut0.SetLutTable(node, bit, table_value);
			lut1.SetLutTable(node, bit, table_value);
		}
	}
	
	// データ設定
	auto in_val = lut0.GetInputValueBuffer();
	lut1.SetInputValueBuffer(in_val);		// 入力バッファ共通化
	auto out_val0 = lut0.GetOutputValueBuffer();
	auto out_val1 = lut1.GetOutputValueBuffer();
	
	for (size_t frame = 0; frame < frame_size; ++frame) {
		for (int node = 0; node < input_node_size; ++node) {
			bool input_value = (rand_bin(mt) != 0);
			in_val.SetBinary(frame, node, input_value);
		}
	}

	// 出力バッファを壊しておく(Debug版だと同じ初期値が埋まるので)
	for (size_t frame = 0; frame < frame_size; ++frame) {
		for (int node = 0; node < output_node_size; ++node) {
			out_val0.SetBinary(frame, node, (rand_bin(mt) != 0));
			out_val1.SetBinary(frame, node, (rand_bin(mt) != 0));
		}
	}

	lut0.Forward();
	lut1.Forward();

	for (size_t frame = 0; frame < frame_size; ++frame) {
		for (int node = 0; node < output_node_size; ++node) {
			EXPECT_EQ(out_val0.GetBinary(frame, node), out_val1.GetBinary(frame, node));
		}
	}
}



TEST(NeuralNetBinaryLut, testNeuralNetBinaryLutFeedback)
{
	const size_t lut_input_size = 6;
	const size_t lut_table_size = (1 << lut_input_size);
	const size_t mux_size   = 3;
	const size_t batch_size = lut_table_size;
	const size_t frame_size = mux_size * batch_size;
	const size_t node_size = lut_table_size;
	bb::NeuralNetBinaryLutN<lut_input_size> lut(lut_input_size, lut_table_size, mux_size, batch_size, 1);
	testSetupLayerBuffer(lut);
	
//	__m256i in[lut_input_size];
//	__m256i out[node_size];
//	NeuralNetBufferAccessorBinary<> accIn(in, frame_size);
//	NeuralNetBufferAccessorBinary<> accOut(out, frame_size);

	auto in_val = lut.GetInputValueBuffer();
	auto out_val = lut.GetOutputValueBuffer();

	for (size_t frame = 0; frame < frame_size; frame++) {
		for (int bit = 0; bit < lut_input_size; bit++) {
			in_val.Set<bool>(frame, bit, (frame & ((size_t)1 << bit)) != 0);
		}
	}

	for (size_t node = 0; node < node_size; node++) {
		for (int i = 0; i < lut_input_size; i++) {
			lut.SetLutInput(node, i, i);
		}
	}

//	lut.SetInputValuePtr(in);
//	lut.SetOutputValuePtr(out);
	lut.Forward();

//	uint64_t outW[node_size];
//	uint64_t lutTable[lut_table_size];

	std::vector<float> vec_loss(frame_size);
	std::vector<float> vec_out(frame_size);
	for (int loop = 0; loop < 1; loop++) {
		do {
			/*
			for (int i = 0; i < lut_table_size; i++) {
				outW[i] = out[i].m256i_u64[0];
			}
			for (size_t node = 0; node < node_size; node++) {
				lutTable[node] = 0;
				for (int i = 0; i < lut_table_size; i++) {
					lutTable[node] |= lut.GetLutTable(node, i) ? ((uint64_t)1 << i) : 0;
				}
			}
			*/

			for (size_t frame = 0; frame < frame_size; frame++) {
				vec_loss[frame] = 0;
				for (size_t node = 0; node < node_size; node++) {
					bool val = out_val.Get<bool>(frame, node);
					if (frame % node_size == node) {
						vec_loss[frame] += !val ? +1.0f : -1.0f;
					}
					else {
						vec_loss[frame] += val ? +0.01f : -0.01f;
					}
				}
			}
		} while (lut.Feedback(vec_loss));
		std::cout << loop << std::endl;
	}

	for (size_t node = 0; node < node_size; node++) {
		for (int i = 0; i < lut_table_size; i++) {
			EXPECT_EQ(node == i, lut.GetLutTable(node, i));
		}
	}
}
#endif


TEST(NeuralNetConvolutionTest, testNeuralNetConvolution)
{
	bb::NeuralNetConvolution<> cnv(1, 3, 3, 1, 2, 2);
	testSetupLayerBuffer(cnv);
	auto in_val = cnv.GetInputValueBuffer();
	auto out_val = cnv.GetOutputValueBuffer();

	EXPECT_EQ(9, cnv.GetInputNodeSize());
	EXPECT_EQ(4, cnv.GetOutputNodeSize());

	in_val.SetReal(0, 3 * 0 + 0, 0.1f);
	in_val.SetReal(0, 3 * 0 + 1, 0.2f);
	in_val.SetReal(0, 3 * 0 + 2, 0.3f);
	in_val.SetReal(0, 3 * 1 + 0, 0.4f);
	in_val.SetReal(0, 3 * 1 + 1, 0.5f);
	in_val.SetReal(0, 3 * 1 + 2, 0.6f);
	in_val.SetReal(0, 3 * 2 + 0, 0.7f);
	in_val.SetReal(0, 3 * 2 + 1, 0.8f);
	in_val.SetReal(0, 3 * 2 + 2, 0.9f);

	cnv.W(0, 0, 0, 0) = 0.1f;
	cnv.W(0, 0, 0, 1) = 0.2f;
	cnv.W(0, 0, 1, 0) = 0.3f;
	cnv.W(0, 0, 1, 1) = 0.4f;

	cnv.b(0) = 0.321f;

	cnv.Forward();

	float exp00 = 0.321f
		+ 0.1f * 0.1f
		+ 0.2f * 0.2f
		+ 0.4f * 0.3f
		+ 0.5f * 0.4f;

	float exp01 = 0.321f
		+ 0.2f * 0.1f
		+ 0.3f * 0.2f
		+ 0.5f * 0.3f
		+ 0.6f * 0.4f;

	float exp10 = 0.321f
		+ 0.4f * 0.1f
		+ 0.5f * 0.2f
		+ 0.7f * 0.3f
		+ 0.8f * 0.4f;

	float exp11 = 0.321f
		+ 0.5f * 0.1f
		+ 0.6f * 0.2f
		+ 0.8f * 0.3f
		+ 0.9f * 0.4f;

	EXPECT_EQ(exp00, out_val.GetReal(0, 0));
	EXPECT_EQ(exp01, out_val.GetReal(0, 1));
	EXPECT_EQ(exp10, out_val.GetReal(0, 2));
	EXPECT_EQ(exp11, out_val.GetReal(0, 3));
}


#if 0

TEST(NeuralNetBufferTest, testNeuralNetBufferTest)
{
	bb::NeuralNetBuffer<> buf(10, 2*3*4, BB_TYPE_REAL32);
	
	for (int i = 0; i < 2 * 3 * 4; ++i) {
		buf.Set<float>(0, i, (float)i);
	}

	buf.SetDimension({ 2, 3, 4 });
	EXPECT_EQ(0, *(float *)buf.GetPtr3(0, 0, 0));
	EXPECT_EQ(1, *(float *)buf.GetPtr3(0, 0, 1));
	EXPECT_EQ(2, *(float *)buf.GetPtr3(0, 1, 0));
	EXPECT_EQ(3, *(float *)buf.GetPtr3(0, 1, 1));
	EXPECT_EQ(4, *(float *)buf.GetPtr3(0, 2, 0));
	EXPECT_EQ(5, *(float *)buf.GetPtr3(0, 2, 1));
	EXPECT_EQ(6, *(float *)buf.GetPtr3(1, 0, 0));
	EXPECT_EQ(7, *(float *)buf.GetPtr3(1, 0, 1));
	EXPECT_EQ(8, *(float *)buf.GetPtr3(1, 1, 0));
	EXPECT_EQ(9, *(float *)buf.GetPtr3(1, 1, 1));
	EXPECT_EQ(10, *(float *)buf.GetPtr3(1, 2, 0));
	EXPECT_EQ(11, *(float *)buf.GetPtr3(1, 2, 1));
	EXPECT_EQ(12, *(float *)buf.GetPtr3(2, 0, 0));
	EXPECT_EQ(13, *(float *)buf.GetPtr3(2, 0, 1));
	EXPECT_EQ(14, *(float *)buf.GetPtr3(2, 1, 0));
	EXPECT_EQ(15, *(float *)buf.GetPtr3(2, 1, 1));
	EXPECT_EQ(16, *(float *)buf.GetPtr3(2, 2, 0));
	EXPECT_EQ(17, *(float *)buf.GetPtr3(2, 2, 1));
	EXPECT_EQ(18, *(float *)buf.GetPtr3(3, 0, 0));
	EXPECT_EQ(19, *(float *)buf.GetPtr3(3, 0, 1));
	EXPECT_EQ(20, *(float *)buf.GetPtr3(3, 1, 0));
	EXPECT_EQ(21, *(float *)buf.GetPtr3(3, 1, 1));
	EXPECT_EQ(22, *(float *)buf.GetPtr3(3, 2, 0));
	EXPECT_EQ(23, *(float *)buf.GetPtr3(3, 2, 1));

	int i = 0;
	buf.ResetPtr();
	while (!buf.IsEnd()) {
		EXPECT_EQ((float)i, *(float *)buf.NextPtr());
		i++;
	}
	EXPECT_EQ(i, 24);


	buf.SetRoi({ 0, 1, 0 });
	EXPECT_EQ(2, *(float *)buf.GetPtr3(0, 0, 0));
	EXPECT_EQ(3, *(float *)buf.GetPtr3(0, 0, 1));
	EXPECT_EQ(4, *(float *)buf.GetPtr3(0, 1, 0));
	EXPECT_EQ(5, *(float *)buf.GetPtr3(0, 1, 1));
	EXPECT_EQ(8, *(float *)buf.GetPtr3(1, 0, 0));
	EXPECT_EQ(9, *(float *)buf.GetPtr3(1, 0, 1));
	EXPECT_EQ(10, *(float *)buf.GetPtr3(1, 1, 0));
	EXPECT_EQ(11, *(float *)buf.GetPtr3(1, 1, 1));
	EXPECT_EQ(14, *(float *)buf.GetPtr3(2, 0, 0));
	EXPECT_EQ(15, *(float *)buf.GetPtr3(2, 0, 1));
	EXPECT_EQ(16, *(float *)buf.GetPtr3(2, 1, 0));
	EXPECT_EQ(17, *(float *)buf.GetPtr3(2, 1, 1));
	EXPECT_EQ(20, *(float *)buf.GetPtr3(3, 0, 0));
	EXPECT_EQ(21, *(float *)buf.GetPtr3(3, 0, 1));
	EXPECT_EQ(22, *(float *)buf.GetPtr3(3, 1, 0));
	EXPECT_EQ(23, *(float *)buf.GetPtr3(3, 1, 1));

	buf.ResetPtr();
	EXPECT_EQ(false, buf.IsEnd());
	EXPECT_EQ(2,  *(float *)buf.NextPtr());
	EXPECT_EQ(false, buf.IsEnd());
	EXPECT_EQ(3,  *(float *)buf.NextPtr());
	EXPECT_EQ(4,  *(float *)buf.NextPtr());
	EXPECT_EQ(5,  *(float *)buf.NextPtr());
	EXPECT_EQ(8,  *(float *)buf.NextPtr());
	EXPECT_EQ(9,  *(float *)buf.NextPtr());
	EXPECT_EQ(10, *(float *)buf.NextPtr());
	EXPECT_EQ(11, *(float *)buf.NextPtr());
	EXPECT_EQ(14, *(float *)buf.NextPtr());
	EXPECT_EQ(15, *(float *)buf.NextPtr());
	EXPECT_EQ(16, *(float *)buf.NextPtr());
	EXPECT_EQ(17, *(float *)buf.NextPtr());
	EXPECT_EQ(20, *(float *)buf.NextPtr());
	EXPECT_EQ(21, *(float *)buf.NextPtr());
	EXPECT_EQ(22, *(float *)buf.NextPtr());
	EXPECT_EQ(false, buf.IsEnd());
	EXPECT_EQ(23, *(float *)buf.NextPtr());
	EXPECT_EQ(true, buf.IsEnd());
	
	buf.SetRoi({ 0, 0, 2 });
	EXPECT_EQ(14, *(float *)buf.GetPtr3(0, 0, 0));
	EXPECT_EQ(15, *(float *)buf.GetPtr3(0, 0, 1));
	EXPECT_EQ(16, *(float *)buf.GetPtr3(0, 1, 0));
	EXPECT_EQ(17, *(float *)buf.GetPtr3(0, 1, 1));
	EXPECT_EQ(20, *(float *)buf.GetPtr3(1, 0, 0));
	EXPECT_EQ(21, *(float *)buf.GetPtr3(1, 0, 1));
	EXPECT_EQ(22, *(float *)buf.GetPtr3(1, 1, 0));
	EXPECT_EQ(23, *(float *)buf.GetPtr3(1, 1, 1));

	EXPECT_EQ(14, *(float *)buf.GetPtr(0));
	EXPECT_EQ(15, *(float *)buf.GetPtr(1));
	EXPECT_EQ(16, *(float *)buf.GetPtr(2));
	EXPECT_EQ(17, *(float *)buf.GetPtr(3));
	EXPECT_EQ(20, *(float *)buf.GetPtr(4));
	EXPECT_EQ(21, *(float *)buf.GetPtr(5));
	EXPECT_EQ(22, *(float *)buf.GetPtr(6));
	EXPECT_EQ(23, *(float *)buf.GetPtr(7));

	buf.ResetPtr();
	EXPECT_EQ(false, buf.IsEnd());
	EXPECT_EQ(14, *(float *)buf.NextPtr());
	EXPECT_EQ(false, buf.IsEnd());
	EXPECT_EQ(15, *(float *)buf.NextPtr());
	EXPECT_EQ(16, *(float *)buf.NextPtr());
	EXPECT_EQ(17, *(float *)buf.NextPtr());
	EXPECT_EQ(20, *(float *)buf.NextPtr());
	EXPECT_EQ(21, *(float *)buf.NextPtr());
	EXPECT_EQ(false, buf.IsEnd());
	EXPECT_EQ(22, *(float *)buf.NextPtr());
	EXPECT_EQ(false, buf.IsEnd());
	EXPECT_EQ(23, *(float *)buf.NextPtr());
	EXPECT_EQ(true, buf.IsEnd());
}


TEST(NeuralNetBufferTest, testNeuralNetBufferTest2)
{
	bb::NeuralNetBuffer<> buf(10, 2 * 3 * 4, BB_TYPE_REAL32);

	for (int i = 0; i < 2 * 3 * 4; ++i) {
		buf.Set<float>(0, i, (float)i);
	}


}


#endif