#include <stdio.h>
#include <iostream>
#include "gtest/gtest.h"
#include "bb/NeuralNetType.h"



TEST(NeuralNetTypeTest, testBinary)
{
	bb::Binary bin_bool0(true);
	bb::Binary bin_bool1(false);
	bb::Binary bin_int0(0);
	bb::Binary bin_int1(1);
	bb::Binary bin_int2(-1);

	EXPECT_EQ((bool)bin_bool0, true);
	EXPECT_EQ((bool)bin_bool1, false);
	EXPECT_EQ((bool)bin_int0, false);
	EXPECT_EQ((bool)bin_int1, true);
	EXPECT_EQ((bool)bin_int2, false);

	// bool
	if (bin_bool0) {
		EXPECT_TRUE(true);
	}
	else {
		EXPECT_TRUE(false);
	}

	if (bin_bool1) {
		EXPECT_TRUE(false);
	}
	else {
		EXPECT_TRUE(true);
	}

	EXPECT_EQ(1, (int)bin_bool0);
	EXPECT_EQ(0, (int)bin_bool1);
	EXPECT_EQ(1, (double)bin_bool0);
	EXPECT_EQ(0, (double)bin_bool1);

	bb::Sign sign0(bin_bool0);
	bb::Sign sign1;
	sign1 = bin_bool1;

	EXPECT_EQ(+1, (int)sign0);
	EXPECT_EQ(-1, (int)sign1);
	EXPECT_EQ(+1, (double)sign0);
	EXPECT_EQ(-1, (double)sign1);

}

