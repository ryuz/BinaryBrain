#include <stdio.h>
#include <iostream>
#include <random>
#include "gtest/gtest.h"

#include "bb/BinaryScaling.h"



TEST(BinaryNormalizationTest, testBinaryNormalization)
{
    int const frame_size = 65536;
    int const node_size  = 8;

    auto scale = bb::BinaryScaling<bb::Bit, float>::Create();
    bb::FrameBuffer x_buf(BB_TYPE_BIT, frame_size, node_size);
    scale->SetInputShape(x_buf.GetShape());
    
    std::mt19937_64                         mt(1);
    std::uniform_real_distribution<float>   dist(0.0f, 1.0f);

    float src[node_size];
    float gain[node_size];
    float offset[node_size];
    float exp[node_size];

    // テストパターン
    src[0] = 0.5f; gain[0] = 1.0f; offset[0] = +0.0f;
    src[1] = 0.4f; gain[1] = 1.0f; offset[1] = +0.2f;
    src[2] = 0.5f; gain[2] = 1.0f; offset[2] = -0.2f;
    src[3] = 0.6f; gain[3] = 1.1f; offset[3] = +0.0f;
    src[4] = 0.4f; gain[4] = 0.7f; offset[4] = +0.0f;
    src[5] = 0.3f; gain[5] = 0.5f; offset[5] = +0.1f;
    src[6] = 0.4f; gain[6] = 1.2f; offset[6] = -0.1f;
    src[7] = 0.5f; gain[7] = 0.8f; offset[7] = +0.2f;
    for (int i = 0; i < node_size; ++i) {
        exp[i] = src[i] * gain[i] + offset[i];
    }
    
    // 入力設定
    for (int node = 0; node < node_size; ++node) {
        for (int frame = 0; frame < frame_size; ++frame) {
            x_buf.SetBit(frame, node, dist(mt) < src[node]);
        }
    }

    // 入力頻度計測
    for (int node = 0; node < node_size; ++node) {
        int sum = 0;
        for (int frame = 0; frame < frame_size; ++frame) {
            if ( x_buf.GetBit(frame, node) ) {
                sum++;
            }
        }
        float mean = (float)sum / (float)frame_size;
        std::cout << "input " << node << " : " << mean << std::endl;
        EXPECT_NEAR(src[node], mean, 0.01);
    }

    // パラメータ設定
    for (int node = 0; node < node_size; ++node) {
        scale->SetParameter(node, gain[node], offset[node]);
    }

    // forward
    auto y_buf = scale->Forward(x_buf);

    // 出力頻度計測
    for (int node = 0; node < node_size; ++node) {
        int sum = 0;
        for (int frame = 0; frame < frame_size; ++frame) {
            if ( y_buf.GetBit(frame, node) ) {
                sum++;
            }
        }
        float mean = (float)sum / (float)frame_size;
        std::cout << "output " << node << " : " << mean << "  (exp:" << exp[node] << ")" << std::endl;
//      EXPECT_NEAR(exp[node], mean, 0.01);
    }

}


