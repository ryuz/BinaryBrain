#include <stdio.h>
#include <iostream>
#include "gtest/gtest.h"
#include "bb/MetricsCategoricalAccuracy.h"



TEST(MetricsCategoricalAccuracyTest, testMetricsCategoricalAccuracyTest)
{
	bb::FrameBuffer y_buf(BB_TYPE_FP32, 2, 3);
	bb::FrameBuffer t_buf(BB_TYPE_FP32, 2, 3);
    
    y_buf.SetFP32(0, 0, 0.2f);
    y_buf.SetFP32(0, 1, 0.4f);
    y_buf.SetFP32(0, 2, 0.1f);

    y_buf.SetFP32(1, 0, 0.9f);
    y_buf.SetFP32(1, 1, 0.1f);
    y_buf.SetFP32(1, 2, 0.5f);

    t_buf.SetFP32(0, 0, 0.0f);
    t_buf.SetFP32(0, 1, 1.0f);
    t_buf.SetFP32(0, 2, 0.0f);
    
    t_buf.SetFP32(1, 0, 0.0f);
    t_buf.SetFP32(1, 1, 0.0f);
    t_buf.SetFP32(1, 2, 1.0f);

    auto accFunc = bb::MetricsCategoricalAccuracy<>::Create();
    accFunc->CalculateMetrics(y_buf, t_buf);

    auto acc = accFunc->GetMetrics();
    EXPECT_FLOAT_EQ(0.5, acc);
//  std::cout << "acc : " << acc << std::endl;
}


