#include <string>
#include <iostream>
#include <fstream>

#include "gtest/gtest.h"

#include "bb/DifferentiableLutN.h"


TEST(DifferentiableLutTest, test_001)
{
    auto lut0 = bb::DifferentiableLutN<6, float>::Create(16);
    bb::FrameBuffer x_buf(1, {16}, BB_TYPE_FP32);
    
    lut0->SetInputShape(x_buf.GetShape());
    
    for (int i = 0; i < 16; ++i) {
        x_buf.SetFP32(0, 0, (float)i/10);
    }
    auto y_buf = lut0->Forward(x_buf);

    bb::FrameBuffer dy_buf(1, {16}, BB_TYPE_FP32);
    for (int i = 0; i < 16; ++i) {
        dy_buf.SetFP32(0, 0, (float)i/100);
    }
    
    lut0->Backward(dy_buf);

    {
        std::ofstream ofs("DifferentiableLutTest001.bb_net", std::ios::binary);
        lut0->DumpObject(ofs);
    }

    auto lut1 = bb::DifferentiableLutN<6, float>::Create(16);
    {
        std::ifstream ifs("DifferentiableLutTest001.bb_net", std::ios::binary);
        lut1->LoadObject(ifs);
    }

    EXPECT_EQ(lut0->GetGamma(),              lut1->GetGamma());
    EXPECT_EQ(lut0->GetBeta(),               lut1->GetBeta());
    EXPECT_EQ(lut0->GetOutputShape(),        lut1->GetOutputShape());
    EXPECT_EQ(lut0->GetInputShape(),         lut1->GetInputShape());
    for ( bb::index_t out_node = 0; out_node < lut0->GetOutputNodeSize(); ++out_node ) {
        EXPECT_EQ(lut0->GetNodeConnectionSize(out_node), lut1->GetNodeConnectionSize(out_node));
        for ( bb::index_t in_index = 0; in_index < lut0->GetNodeConnectionSize(out_node); ++in_index ) {
            EXPECT_EQ(lut0->GetNodeConnectionIndex(out_node, in_index), lut1->GetNodeConnectionIndex(out_node, in_index));
        }
    }



}


