#include <stdio.h>
#include <random>
#include <iostream>
#include "gtest/gtest.h"

#include "bb/FrameBuffer.h"


#if BB_WITH_CEREAL
#include "cereal/types/array.hpp"
#include "cereal/types/string.hpp"
#include "cereal/types/vector.hpp"
#include "cereal/archives/json.hpp"
#endif


TEST(FrameBufferTest, FrameBuffer_SetGet)
{
    bb::FrameBuffer buf(BB_TYPE_FP32, 2, 3);

    buf.Set<float, float>(0, 0, 1.0f);
    buf.Set<float, float>(0, 1, 2.0f);
    buf.Set<float, float>(0, 2, 3.0f);
    buf.Set<float, float>(1, 0, 4.0f);
    buf.Set<float, float>(1, 1, 5.0f);
    buf.Set<float, float>(1, 2, 6.0f);

    EXPECT_EQ(1.0f, (buf.Get<float, float>(0, 0)));
    EXPECT_EQ(2.0f, (buf.Get<float, float>(0, 1)));
    EXPECT_EQ(3.0f, (buf.Get<float, float>(0, 2)));
    EXPECT_EQ(4.0f, (buf.Get<float, float>(1, 0)));
    EXPECT_EQ(5.0f, (buf.Get<float, float>(1, 1)));
    EXPECT_EQ(6.0f, (buf.Get<float, float>(1, 2)));


    // 型指定
    buf.SetFP64(0, 0, 8.1);
    buf.SetFP64(0, 1, 7.2);
    buf.SetFP64(0, 2, 6.3);
    buf.SetFP64(1, 0, 5.4);
    buf.SetFP64(1, 1, 4.5);
    buf.SetFP64(1, 2, 3.6);
    
    EXPECT_EQ(8, buf.GetINT32(0, 0));
    EXPECT_EQ(7, buf.GetINT32(0, 1));
    EXPECT_EQ(6, buf.GetINT32(0, 2));
    EXPECT_EQ(5, buf.GetINT32(1, 0));
    EXPECT_EQ(4, buf.GetINT32(1, 1));
    EXPECT_EQ(3, buf.GetINT32(1, 2));

    EXPECT_EQ(8.1f, buf.GetFP32(0, 0));
    EXPECT_EQ(7.2f, buf.GetFP32(0, 1));
    EXPECT_EQ(6.3f, buf.GetFP32(0, 2));
    EXPECT_EQ(5.4f, buf.GetFP32(1, 0));
    EXPECT_EQ(4.5f, buf.GetFP32(1, 1));
    EXPECT_EQ(3.6f, buf.GetFP32(1, 2));


    // clone
    buf.Set<float, float>(0, 0, 11.0f);
    buf.Set<float, float>(0, 1, 12.0f);
    buf.Set<float, float>(0, 2, 13.0f);
    buf.Set<float, float>(1, 0, 14.0f);
    buf.Set<float, float>(1, 1, 15.0f);
    buf.Set<float, float>(1, 2, 16.0f);

    auto buf1 = buf;
    auto buf2 = buf.Clone();

    buf.Set<float, float>(0, 0, 21.0f);
    buf.Set<float, float>(0, 1, 22.0f);
    buf.Set<float, float>(0, 2, 23.0f);
    buf.Set<float, float>(1, 0, 24.0f);
    buf.Set<float, float>(1, 1, 25.0f);
    buf.Set<float, float>(1, 2, 26.0f);

    // ポインタコピーは新しい値
    EXPECT_EQ(21.0f, (buf1.Get<float, float>(0, 0)));
    EXPECT_EQ(22.0f, (buf1.Get<float, float>(0, 1)));
    EXPECT_EQ(23.0f, (buf1.Get<float, float>(0, 2)));
    EXPECT_EQ(24.0f, (buf1.Get<float, float>(1, 0)));
    EXPECT_EQ(25.0f, (buf1.Get<float, float>(1, 1)));
    EXPECT_EQ(26.0f, (buf1.Get<float, float>(1, 2)));

    // クローンは古い値を維持
    EXPECT_EQ(11.0f, (buf2.Get<float, float>(0, 0)));
    EXPECT_EQ(12.0f, (buf2.Get<float, float>(0, 1)));
    EXPECT_EQ(13.0f, (buf2.Get<float, float>(0, 2)));
    EXPECT_EQ(14.0f, (buf2.Get<float, float>(1, 0)));
    EXPECT_EQ(15.0f, (buf2.Get<float, float>(1, 1)));
    EXPECT_EQ(16.0f, (buf2.Get<float, float>(1, 2)));
}


TEST(FrameBufferTest, FrameBuffer_IsDeviceAvailable)
{
    bb::FrameBuffer buf_h(BB_TYPE_FP32, 2, 3, true);
    bb::FrameBuffer buf_d(BB_TYPE_FP32, 2, 3, false);
#if BB_WITH_CUDA
    int dev_count = 0;
    auto status = cudaGetDeviceCount(&dev_count);
    if ( status == cudaSuccess && dev_count > 0 ) {
        EXPECT_FALSE(buf_h.IsDeviceAvailable());
        EXPECT_TRUE(buf_d.IsDeviceAvailable());
    }
    else {
        EXPECT_FALSE(buf_h.IsDeviceAvailable());
        EXPECT_FALSE(buf_d.IsDeviceAvailable());
    }
#else
    EXPECT_FALSE(buf_h.IsDeviceAvailable());
    EXPECT_FALSE(buf_d.IsDeviceAvailable());
#endif
}


TEST(FrameBufferTest, FrameBuffer_Range)
{
    bb::index_t const frame_size = 32;
    bb::index_t const node_size  = 3;

    bb::FrameBuffer buf(BB_TYPE_INT32, frame_size, node_size);

    for ( bb::index_t frame = 0; frame < frame_size; ++frame ) {
        for ( bb::index_t node = 0; node < node_size; ++node ) {
            buf.SetINT32(frame, node, (int)(frame*100 + node));
        }
    }

    for ( bb::index_t frame = 0; frame < frame_size; ++frame ) {
        for ( bb::index_t node = 0; node < node_size; ++node ) {
//          std::cout << buf.GetINT32(frame, node) << std::endl;
            EXPECT_EQ(buf.GetINT32(frame, node), (int)(frame*100 + node));
        }
    }

    auto buf2 = buf.GetRange(4, 2);
    EXPECT_EQ(buf2.GetFrameSize(), 2);
    for ( bb::index_t frame = 0; frame < buf2.GetFrameSize(); ++frame ) {
        for ( bb::index_t node = 0; node < buf2.GetNodeSize(); ++node ) {
            EXPECT_EQ(buf2.GetINT32(frame, node), (int)((frame+4)*100 + node));
        }
    }
}



TEST(FrameBufferTest, FrameBuffer_SetGetTensor)
{
    bb::index_t const frame_size = 32;
    bb::index_t const l = 3;
    bb::index_t const m = 4;
    bb::index_t const n = 5;

    bb::FrameBuffer buf(BB_TYPE_INT64, frame_size, {l, m, n});

    for ( bb::index_t frame = 0; frame < frame_size; ++frame ) {
        for ( bb::index_t i = 0; i < n; ++i ) {
            for ( bb::index_t j = 0; i < m; ++i ) {
                for ( bb::index_t k = 0; i < l; ++i ) {
                    buf.SetINT64(frame, {k, j, i}, (int)(frame*1000000 + i*10000 + j * 100 + k));
                }
            }
        }
    }

    for ( bb::index_t frame = 0; frame < frame_size; ++frame ) {
        for ( bb::index_t i = 0; i < n; ++i ) {
            for ( bb::index_t j = 0; i < m; ++i ) {
                for ( bb::index_t k = 0; i < l; ++i ) {
                    EXPECT_EQ(buf.GetINT64(frame, {k, j, i}), (int)(frame*1000000 + i*10000 + j * 100 + k));
                }
            }
        }
    }
}


TEST(FrameBufferTest, FrameBuffer_SetGetTensorBit)
{
    bb::index_t const frame_size = 32;
    bb::index_t const l = 3;
    bb::index_t const m = 4;
    bb::index_t const n = 5;

    bb::FrameBuffer buf(BB_TYPE_BIT, frame_size, {l, m, n});

    for ( bb::index_t frame = 0; frame < frame_size; ++frame ) {
        for ( bb::index_t i = 0; i < n; ++i ) {
            for ( bb::index_t j = 0; i < m; ++i ) {
                for ( bb::index_t k = 0; i < l; ++i ) {
                    buf.Set<bb::Bit, bb::Bit>(frame, {k, j, i}, (bb::Bit)(frame ^ i ^ j ^ k));
                }
            }
        }
    }

    for ( bb::index_t frame = 0; frame < frame_size; ++frame ) {
        for ( bb::index_t i = 0; i < n; ++i ) {
            for ( bb::index_t j = 0; i < m; ++i ) {
                for ( bb::index_t k = 0; i < l; ++i ) {
                    EXPECT_EQ((buf.Get<bb::Bit, bb::Bit>(frame, {k, j, i})), (bb::Bit)(frame ^ i ^ j ^ k));
                }
            }
        }
    }
}



TEST(FrameBufferTest, FrameBuffer_SetVector)
{
    bb::index_t const frame_size = 32;
    bb::index_t const node_size = 12;

    bb::FrameBuffer buf(BB_TYPE_FP32, frame_size, node_size);

    std::vector< std::vector<float> > vec(frame_size, std::vector<float>(node_size));
    
    for ( bb::index_t frame = 0; frame < frame_size; ++frame ) {
        for ( bb::index_t node = 0; node < node_size; ++node ) {
            vec[frame][node] = (float)(frame * 1000 + node);
        }
    }

    buf.SetVector(vec);

    for ( bb::index_t frame = 0; frame < frame_size; ++frame ) {
        for ( bb::index_t node = 0; node < node_size; ++node ) {
            EXPECT_EQ((buf.Get<float, float>(frame, node)), (float)(frame * 1000 + node));
        }
    }
}



TEST(FrameBufferTest, testFrameBuffer_Json)
{
    bb::index_t const frame_size = 32;
    bb::index_t const node_size = 12;

    std::vector< std::vector<float> > vec(frame_size, std::vector<float>(node_size));
    
    for ( bb::index_t frame = 0; frame < frame_size; ++frame ) {
        for ( bb::index_t node = 0; node < node_size; ++node ) {
            vec[frame][node] = (float)(frame * 1000 + node);
        }
    }

    // save
    {
        bb::FrameBuffer buf(BB_TYPE_FP32, frame_size, node_size);
        buf.SetVector(vec);
        {
            std::ofstream ofs("FrameBufferTest.bin", std::ios::binary);
            buf.Save(ofs);
        }

#ifdef BB_WITH_CEREAL
        {
            std::ofstream ofs("FrameBufferTest.json");
            cereal::JSONOutputArchive ar(ofs);
            ar(cereal::make_nvp("FrameBuffer", buf));
        }
#endif
    }


    // load
    {
        bb::FrameBuffer buf(BB_TYPE_FP64, 1, 2);
        std::ifstream ifs("FrameBufferTest.bin", std::ios::binary);
        buf.Load(ifs);

        for ( bb::index_t frame = 0; frame < frame_size; ++frame ) {
            for ( bb::index_t node = 0; node < node_size; ++node ) {
                EXPECT_EQ((buf.Get<float, float>(frame, node)), (float)(frame * 1000 + node));
            }
        }
    }

#ifdef BB_WITH_CEREAL
    {
        bb::FrameBuffer buf(BB_TYPE_FP64, 1, 2);
        std::ifstream ifs("FrameBufferTest.json");
        cereal::JSONInputArchive ar(ifs);
        ar(cereal::make_nvp("FrameBuffer", buf));

        for ( bb::index_t frame = 0; frame < frame_size; ++frame ) {
            for ( bb::index_t node = 0; node < node_size; ++node ) {
                EXPECT_EQ((buf.Get<float, float>(frame, node)), (float)(frame * 1000 + node));
            }
        }
    }
#endif

}



TEST(FrameBufferTest, FrameBuffer_CopyTo_fp32)
{
    int const src_frame_size = 5;
    int const src_node_size  = 7;
    bb::FrameBuffer src_buf(BB_TYPE_FP32, src_frame_size, src_node_size);
    for ( bb::index_t frame = 0; frame < src_frame_size; ++frame ) {
        for ( bb::index_t node = 0; node < src_node_size; ++node ) {
            src_buf.SetFP32(frame, node, (float)(frame*100 + node));
        }
    }
    
    bb::FrameBuffer dst0_buf(BB_TYPE_FP32, 6, 5);
    for ( bb::index_t frame = 0; frame < 6; ++frame ) {
        for ( bb::index_t node = 0; node < 5; ++node ) {
            dst0_buf.SetFP32(frame, node, (float)(1000 + frame*100 + node));
        }
    }

    src_buf.CopyTo(dst0_buf, 2, 2, 1, 2, 3, 2);

    EXPECT_EQ(1101, dst0_buf.GetFP32(1, 1));
    EXPECT_EQ( 203, dst0_buf.GetFP32(1, 2));
    EXPECT_EQ( 204, dst0_buf.GetFP32(1, 3));
    EXPECT_EQ(1104, dst0_buf.GetFP32(1, 4));

    EXPECT_EQ(1201, dst0_buf.GetFP32(2, 1));
    EXPECT_EQ( 303, dst0_buf.GetFP32(2, 2));
    EXPECT_EQ( 304, dst0_buf.GetFP32(2, 3));
    EXPECT_EQ(1204, dst0_buf.GetFP32(2, 4));

    EXPECT_EQ(1002, dst0_buf.GetFP32(0, 2));
    EXPECT_EQ( 203, dst0_buf.GetFP32(1, 2));
    EXPECT_EQ( 303, dst0_buf.GetFP32(2, 2));
    EXPECT_EQ(1302, dst0_buf.GetFP32(3, 2));

    EXPECT_EQ(1003, dst0_buf.GetFP32(0, 3));
    EXPECT_EQ( 204, dst0_buf.GetFP32(1, 3));
    EXPECT_EQ( 304, dst0_buf.GetFP32(2, 3));
    EXPECT_EQ(1303, dst0_buf.GetFP32(3, 3));
}


TEST(FrameBufferTest, FrameBuffer_CopyTo_bit)
{
    int const src_frame_size = 32*7;
    int const src_node_size  = 7;

    std::mt19937_64 mt(1);

    bb::Bit src_tbl[src_frame_size][src_node_size];
    bb::FrameBuffer src_buf(BB_TYPE_FP32, src_frame_size, src_node_size);
    for ( bb::index_t frame = 0; frame < src_frame_size; ++frame ) {
        for ( bb::index_t node = 0; node < src_node_size; ++node ) {
            src_tbl[frame][node] = mt() % 2;
            src_buf.SetBit(frame, node, src_tbl[frame][node]);
        }
    }
    
    int const dst_frame_size = 32*14;
    int const dst_node_size  = 8;

    bb::Bit dst_tbl[dst_frame_size][dst_node_size];
    bb::FrameBuffer dst0_buf(BB_TYPE_FP32, dst_frame_size, dst_node_size);
    for ( bb::index_t frame = 0; frame < dst_frame_size; ++frame ) {
        for ( bb::index_t node = 0; node < dst_node_size; ++node ) {
            dst_tbl[frame][node] = mt() % 2;
            dst0_buf.SetBit(frame, node, dst_tbl[frame][node]);
        }
    }

    src_buf.CopyTo
        (
            dst0_buf,
            32*3, 32*2, 32,
            2,       3,  2
        );

    EXPECT_EQ(dst_tbl[32*1][0], dst0_buf.GetBit(32, 0));
    EXPECT_EQ(dst_tbl[32*1][1], dst0_buf.GetBit(32, 1));
    EXPECT_EQ(src_tbl[32*2][3], dst0_buf.GetBit(32, 2));
    EXPECT_EQ(src_tbl[32*2][4], dst0_buf.GetBit(32, 3));
    EXPECT_EQ(dst_tbl[32*1][4], dst0_buf.GetBit(32, 4));
    EXPECT_EQ(dst_tbl[32*1][5], dst0_buf.GetBit(32, 5));
    EXPECT_EQ(dst_tbl[32*1][6], dst0_buf.GetBit(32, 6));
    EXPECT_EQ(dst_tbl[32*1][7], dst0_buf.GetBit(32, 7));

    EXPECT_EQ(dst_tbl[32*1+31][0], dst0_buf.GetBit(32+31, 0));
    EXPECT_EQ(dst_tbl[32*1+31][1], dst0_buf.GetBit(32+31, 1));
    EXPECT_EQ(src_tbl[32*2+31][3], dst0_buf.GetBit(32+31, 2));
    EXPECT_EQ(src_tbl[32*2+31][4], dst0_buf.GetBit(32+31, 3));
    EXPECT_EQ(dst_tbl[32*1+31][4], dst0_buf.GetBit(32+31, 4));
    EXPECT_EQ(dst_tbl[32*1+31][5], dst0_buf.GetBit(32+31, 5));
    EXPECT_EQ(dst_tbl[32*1+31][6], dst0_buf.GetBit(32+31, 6));
    EXPECT_EQ(dst_tbl[32*1+31][7], dst0_buf.GetBit(32+31, 7));

    EXPECT_EQ(dst_tbl[32*1 -4][2], dst0_buf.GetBit(32 -4, 2));
    EXPECT_EQ(dst_tbl[32*1 -3][2], dst0_buf.GetBit(32 -3, 2));
    EXPECT_EQ(dst_tbl[32*1 -2][2], dst0_buf.GetBit(32 -2, 2));
    EXPECT_EQ(dst_tbl[32*1 -1][2], dst0_buf.GetBit(32 -1, 2));
    EXPECT_EQ(src_tbl[32*2 +0][3], dst0_buf.GetBit(32 +0, 2));
    EXPECT_EQ(src_tbl[32*2 +1][3], dst0_buf.GetBit(32 +1, 2));
    EXPECT_EQ(src_tbl[32*2 +2][3], dst0_buf.GetBit(32 +2, 2));
    EXPECT_EQ(src_tbl[32*2 +3][3], dst0_buf.GetBit(32 +3, 2));
    EXPECT_EQ(src_tbl[32*2 +4][3], dst0_buf.GetBit(32 +4, 2));
    EXPECT_EQ(src_tbl[32*2 +5][3], dst0_buf.GetBit(32 +5, 2));
    EXPECT_EQ(src_tbl[32*2 +6][3], dst0_buf.GetBit(32 +6, 2));
    EXPECT_EQ(src_tbl[32*2+90][3], dst0_buf.GetBit(32+90, 2));
    EXPECT_EQ(src_tbl[32*2+91][3], dst0_buf.GetBit(32+91, 2));
    EXPECT_EQ(src_tbl[32*2+92][3], dst0_buf.GetBit(32+92, 2));
    EXPECT_EQ(src_tbl[32*2+93][3], dst0_buf.GetBit(32+93, 2));
    EXPECT_EQ(src_tbl[32*2+94][3], dst0_buf.GetBit(32+94, 2));
    EXPECT_EQ(src_tbl[32*2+95][3], dst0_buf.GetBit(32+95, 2));
    EXPECT_EQ(dst_tbl[32*1+96][2], dst0_buf.GetBit(32+96, 2));
    EXPECT_EQ(dst_tbl[32*1+97][2], dst0_buf.GetBit(32+97, 2));
    EXPECT_EQ(dst_tbl[32*1+98][2], dst0_buf.GetBit(32+98, 2));
    EXPECT_EQ(dst_tbl[32*1+99][2], dst0_buf.GetBit(32+99, 2));

    EXPECT_EQ(dst_tbl[32*1 -4][3], dst0_buf.GetBit(32 -4, 3));
    EXPECT_EQ(dst_tbl[32*1 -3][3], dst0_buf.GetBit(32 -3, 3));
    EXPECT_EQ(dst_tbl[32*1 -2][3], dst0_buf.GetBit(32 -2, 3));
    EXPECT_EQ(dst_tbl[32*1 -1][3], dst0_buf.GetBit(32 -1, 3));
    EXPECT_EQ(src_tbl[32*2 +0][4], dst0_buf.GetBit(32 +0, 3));
    EXPECT_EQ(src_tbl[32*2 +1][4], dst0_buf.GetBit(32 +1, 3));
    EXPECT_EQ(src_tbl[32*2 +2][4], dst0_buf.GetBit(32 +2, 3));
    EXPECT_EQ(src_tbl[32*2 +3][4], dst0_buf.GetBit(32 +3, 3));
    EXPECT_EQ(src_tbl[32*2 +4][4], dst0_buf.GetBit(32 +4, 3));
    EXPECT_EQ(src_tbl[32*2 +5][4], dst0_buf.GetBit(32 +5, 3));
    EXPECT_EQ(src_tbl[32*2 +6][4], dst0_buf.GetBit(32 +6, 3));
    EXPECT_EQ(src_tbl[32*2+90][4], dst0_buf.GetBit(32+90, 3));
    EXPECT_EQ(src_tbl[32*2+91][4], dst0_buf.GetBit(32+91, 3));
    EXPECT_EQ(src_tbl[32*2+92][4], dst0_buf.GetBit(32+92, 3));
    EXPECT_EQ(src_tbl[32*2+93][4], dst0_buf.GetBit(32+93, 3));
    EXPECT_EQ(src_tbl[32*2+94][4], dst0_buf.GetBit(32+94, 3));
    EXPECT_EQ(src_tbl[32*2+95][4], dst0_buf.GetBit(32+95, 3));
    EXPECT_EQ(dst_tbl[32*1+96][3], dst0_buf.GetBit(32+96, 3));
    EXPECT_EQ(dst_tbl[32*1+97][3], dst0_buf.GetBit(32+97, 3));
    EXPECT_EQ(dst_tbl[32*1+98][3], dst0_buf.GetBit(32+98, 3));
    EXPECT_EQ(dst_tbl[32*1+99][3], dst0_buf.GetBit(32+99, 3));
}
