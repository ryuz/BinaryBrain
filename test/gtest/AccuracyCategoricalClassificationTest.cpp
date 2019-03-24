#include <stdio.h>
#include <iostream>
#include "gtest/gtest.h"
#include "bb/AccuracyCategoricalClassification.h"



TEST(AccuracyCategoricalClassificationTest, testAccuracyCategoricalClassification)
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

    auto accFunc = bb::AccuracyCategoricalClassification<>::Create();
    accFunc->CalculateAccuracy(y_buf, t_buf);

    std::cout << "acc : " << accFunc->GetAccuracy() << std::endl;
}


