#include <stdio.h>
#include <iostream>
#include "gtest/gtest.h"
#include "NeuralNetAffine.h"

/*
int main()
{
	NeuralNetAffine<> nna();

	return 0;
}
*/

#if 1
//namespace {

//	class NeuralNetAffineTest : public ::testing::Test {};

	TEST(NeuralNetAffineTest, test00_AffineForward)
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


	TEST(NeuralNetAffineTest, test00_AffineForwardBatch)
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

	TEST(NeuralNetAffineTest, test01)
	{
		EXPECT_EQ(true, true);
		EXPECT_EQ(false, false);
	}

//} // namespace

#endif
