#include <stdio.h>
#include <iostream>
#include "gtest/gtest.h"
#include "NeuralNetAffine.h"
#include "NeuralNetSigmoid.h"



TEST(NeuralNetAffineTest, AffineForward)
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


TEST(NeuralNetAffineTest, AffineForwardBatch)
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


TEST(NeuralNetSigmoidTest, testSigmoidForward)
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


TEST(NeuralNetSigmoidTest, testSigmoidForwardBatch)
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

