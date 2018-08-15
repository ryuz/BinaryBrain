#include <stdio.h>
#include <iostream>
#include "gtest/gtest.h"
#include "NeuralNetAffine.h"
#include "NeuralNetSigmoid.h"
#include "NeuralNetSoftmax.h"



TEST(NeuralNetAffineTest, testAffine)
{
	NeuralNetAffine<> affine(2, 3);

	float in[2] = { 1, 2 };
	float out[3];

	affine.SetInputValuePtr(in);
	affine.SetOutputValuePtr(out);

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
	EXPECT_EQ(1 * 1 + 2 * 2 + 1000, out[0]);
	EXPECT_EQ(1 * 10 + 2 * 20 + 2000, out[1]);
	EXPECT_EQ(1 * 100 + 2 * 200 + 3000, out[2]);

	float outError[3] = { 998, 2042, 3491 };
	float inError[2];
	affine.SetOutputErrorPtr(outError);
	affine.SetInputErrorPtr(inError);
	affine.Backward();
	EXPECT_EQ(370518, inError[0]);
	EXPECT_EQ(741036, inError[1]);

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
	NeuralNetAffine<> affine(2, 3, 2);

	float in[2*2] = { 1, 3, 2, 4 };
	float out[3*2];

	affine.SetInputValuePtr(in);
	affine.SetOutputValuePtr(out);

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
		
	EXPECT_EQ(1 * 1 + 2 * 2 + 1000, out[0]);
	EXPECT_EQ(1 * 10 + 2 * 20 + 2000, out[2]);
	EXPECT_EQ(1 * 100 + 2 * 200 + 3000, out[4]);

	EXPECT_EQ(3 * 1 + 4 * 2 + 1000, out[1]);
	EXPECT_EQ(3 * 10 + 4 * 20 + 2000, out[3]);
	EXPECT_EQ(3 * 100 + 4 * 200 + 3000, out[5]);

	float outError[3 * 2] = { 998, 1004, 2042, 2102, 3491, 4091 };
	float inError[2 * 2];
	affine.SetOutputErrorPtr(outError);
	affine.SetInputErrorPtr(inError);
	affine.Backward();
	EXPECT_EQ(370518, inError[0]);
	EXPECT_EQ(741036, inError[2]);
	EXPECT_EQ(431124, inError[1]);
	EXPECT_EQ(862248, inError[3]);
}


TEST(NeuralNetSigmoidTest, testSigmoid)
{
	NeuralNetSigmoid<> sigmoid(2);
	float in[2] = { 1, 2};
	float out[2];

	sigmoid.SetInputValuePtr(in);
	sigmoid.SetOutputValuePtr(out);

	sigmoid.Forward();
	EXPECT_EQ(1.0f / (1.0f + exp(-1.0f)), out[0]);
	EXPECT_EQ(1.0f / (1.0f + exp(-2.0f)), out[1]);

	float outError[2] = { 2, 3 };
	float inError[2];
	sigmoid.SetOutputErrorPtr(outError);
	sigmoid.SetInputErrorPtr(inError);
	sigmoid.Backward();

	EXPECT_EQ(outError[0] * (1.0f - out[0]) * out[0], inError[0]);
	EXPECT_EQ(outError[1] * (1.0f - out[1]) * out[1], inError[1]);
}


TEST(NeuralNetSigmoidTest, testSigmoidBatch)
{
	NeuralNetSigmoid<> sigmoid(2, 2);
	float in[2*2] = { 1, 2, 3, 4 };
	float out[2 * 2];

	sigmoid.SetInputValuePtr(in);
	sigmoid.SetOutputValuePtr(out);

	sigmoid.Forward();
	EXPECT_EQ(1.0f / (1.0f + exp(-1.0f)), out[0]);
	EXPECT_EQ(1.0f / (1.0f + exp(-2.0f)), out[1]);
	EXPECT_EQ(1.0f / (1.0f + exp(-3.0f)), out[2]);
	EXPECT_EQ(1.0f / (1.0f + exp(-4.0f)), out[3]);

	float outError[2*2] = { 2, 3, 4, -5 };
	float inError[2*2];
	sigmoid.SetOutputErrorPtr(outError);
	sigmoid.SetInputErrorPtr(inError);
	sigmoid.Backward();

	EXPECT_EQ(outError[0] * (1.0f - out[0]) * out[0], inError[0]);
	EXPECT_EQ(outError[1] * (1.0f - out[1]) * out[1], inError[1]);
	EXPECT_EQ(outError[2] * (1.0f - out[2]) * out[2], inError[2]);
	EXPECT_EQ(outError[3] * (1.0f - out[3]) * out[3], inError[3]);
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
	NeuralNetSoftmax<> softmax(3);
	float in[3];
	float out[3];

	in[0] = m_src(0, 0);
	in[1] = m_src(0, 1);
	in[2] = m_src(0, 2);

	softmax.SetInputValuePtr(in);
	softmax.SetOutputValuePtr(out);

	softmax.Forward();
	EXPECT_EQ(m_softmax(0, 0), out[0]);
	EXPECT_EQ(m_softmax(0, 1), out[1]);
	EXPECT_EQ(m_softmax(0, 2), out[2]);

	float outError[3] = { 2, 3, 4 };
	float inError[3];
	softmax.SetOutputErrorPtr(outError);
	softmax.SetInputErrorPtr(inError);
	softmax.Backward();

	EXPECT_EQ(2, inError[0]);
	EXPECT_EQ(3, inError[1]);
	EXPECT_EQ(4, inError[2]);
}


TEST_F(NeuralNetSoftmaxTest, testSoftmaxBatch)
{
	NeuralNetSoftmax<> softmax(3, 2);
	float in[3*2];
	float out[3*2];

	in[0] = m_src(0, 0);
	in[1] = m_src(1, 0);
	in[2] = m_src(0, 1);
	in[3] = m_src(1, 1);
	in[4] = m_src(0, 2);
	in[5] = m_src(1, 2);

	softmax.SetInputValuePtr(in);
	softmax.SetOutputValuePtr(out);

	softmax.Forward();
	EXPECT_EQ(m_softmax(0, 0), out[0]);
	EXPECT_EQ(m_softmax(1, 0), out[1]);
	EXPECT_EQ(m_softmax(0, 1), out[2]);
	EXPECT_EQ(m_softmax(1, 1), out[3]);
	EXPECT_EQ(m_softmax(0, 2), out[4]);
	EXPECT_EQ(m_softmax(1, 2), out[5]);

	float outError[3*2] = { 2, 3, 4, 5, 6, 7 };
	float inError[3 * 2];
	softmax.SetOutputErrorPtr(outError);
	softmax.SetInputErrorPtr(inError);
	softmax.Backward();

	EXPECT_EQ(2, inError[0]);
	EXPECT_EQ(3, inError[1]);
	EXPECT_EQ(4, inError[2]);
	EXPECT_EQ(5, inError[3]);
	EXPECT_EQ(6, inError[4]);
	EXPECT_EQ(7, inError[5]);
}

