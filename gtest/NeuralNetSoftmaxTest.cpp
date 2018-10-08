#include <stdio.h>
#include <iostream>
#include "gtest/gtest.h"
#include "bb/NeuralNetSoftmax.h"


inline void testSetupLayerBuffer(bb::NeuralNetLayer<>& net)
{
	net.SetInputSignalBuffer (net.CreateInputSignalBuffer());
	net.SetInputErrorBuffer (net.CreateInputErrorBuffer());
	net.SetOutputSignalBuffer(net.CreateOutputSignalBuffer());
	net.SetOutputErrorBuffer(net.CreateOutputErrorBuffer());
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

	auto in_val = softmax.GetInputSignalBuffer();
	auto out_val = softmax.GetOutputSignalBuffer();
	in_val.SetReal(0, 0, m_src(0, 0));
	in_val.SetReal(0, 1, m_src(0, 1));
	in_val.SetReal(0, 2, m_src(0, 2));

	softmax.Forward();

	EXPECT_EQ(m_softmax(0, 0), out_val.GetReal(0, 0));
	EXPECT_EQ(m_softmax(0, 1), out_val.GetReal(0, 1));
	EXPECT_EQ(m_softmax(0, 2), out_val.GetReal(0, 2));

	auto out_err = softmax.GetOutputErrorBuffer();
	auto in_err = softmax.GetInputErrorBuffer();
	out_err.SetReal(0, 0, 2);
	out_err.SetReal(0, 1, 3);
	out_err.SetReal(0, 2, 4);

	softmax.Backward();

	EXPECT_EQ(2, in_err.GetReal(0, 0));
	EXPECT_EQ(3, in_err.GetReal(0, 1));
	EXPECT_EQ(4, in_err.GetReal(0, 2));
}


TEST_F(NeuralNetSoftmaxTest, testSoftmaxBatch)
{
	bb::NeuralNetSoftmax<> softmax(3);
	softmax.SetBatchSize(2);

	testSetupLayerBuffer(softmax);
	
	auto in_val = softmax.GetInputSignalBuffer();
	auto out_val = softmax.GetOutputSignalBuffer();
	in_val.SetReal(0, 0, m_src(0, 0));
	in_val.SetReal(1, 0, m_src(1, 0));
	in_val.SetReal(0, 1, m_src(0, 1));
	in_val.SetReal(1, 1, m_src(1, 1));
	in_val.SetReal(0, 2, m_src(0, 2));
	in_val.SetReal(1, 2, m_src(1, 2));

	softmax.Forward();

	EXPECT_EQ(m_softmax(0, 0), out_val.GetReal(0, 0));
	EXPECT_EQ(m_softmax(1, 0), out_val.GetReal(1, 0));
	EXPECT_EQ(m_softmax(0, 1), out_val.GetReal(0, 1));
	EXPECT_EQ(m_softmax(1, 1), out_val.GetReal(1, 1));
	EXPECT_EQ(m_softmax(0, 2), out_val.GetReal(0, 2));
	EXPECT_EQ(m_softmax(1, 2), out_val.GetReal(1, 2));

	auto out_err = softmax.GetOutputErrorBuffer();
	auto in_err = softmax.GetInputErrorBuffer();
	out_err.SetReal(0, 0, 2);
	out_err.SetReal(1, 0, 3);
	out_err.SetReal(0, 1, 4);
	out_err.SetReal(1, 1, 5);
	out_err.SetReal(0, 2, 6);
	out_err.SetReal(1, 2, 7);

	softmax.Backward();

	EXPECT_EQ(2, in_err.GetReal(0, 0));
	EXPECT_EQ(3, in_err.GetReal(1, 0));
	EXPECT_EQ(4, in_err.GetReal(0, 1));
	EXPECT_EQ(5, in_err.GetReal(1, 1));
	EXPECT_EQ(6, in_err.GetReal(0, 2));
	EXPECT_EQ(7, in_err.GetReal(1, 2));
}


