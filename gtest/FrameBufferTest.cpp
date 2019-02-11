#include <stdio.h>
#include <random>
#include <iostream>
#include "gtest/gtest.h"

#include "bb/FrameBuffer.h"


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
    EXPECT_FALSE(buf_h.IsDeviceAvailable());
    EXPECT_TRUE(buf_d.IsDeviceAvailable());
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



#if 0

TEST(NeuralNetBufferTest, testNeuralNetBufferTest)
{
	bb::NeuralNetBuffer<> buf(10, 2 * 3 * 4, BB_TYPE_REAL32);

	for (int i = 0; i < 2 * 3 * 4; ++i) {
		buf.Set<float>(0, i, (float)i);
	}

	// 多次元配列構成
	buf.SetDimensions({ 2, 3, 4 });
	EXPECT_EQ(0, *(float *)buf.GetPtr3(0, 0, 0));
	EXPECT_EQ(1, *(float *)buf.GetPtr3(0, 0, 1));
	EXPECT_EQ(2, *(float *)buf.GetPtr3(0, 1, 0));
	EXPECT_EQ(3, *(float *)buf.GetPtr3(0, 1, 1));
	EXPECT_EQ(4, *(float *)buf.GetPtr3(0, 2, 0));
	EXPECT_EQ(5, *(float *)buf.GetPtr3(0, 2, 1));
	EXPECT_EQ(6, *(float *)buf.GetPtr3(1, 0, 0));
	EXPECT_EQ(7, *(float *)buf.GetPtr3(1, 0, 1));
	EXPECT_EQ(8, *(float *)buf.GetPtr3(1, 1, 0));
	EXPECT_EQ(9, *(float *)buf.GetPtr3(1, 1, 1));
	EXPECT_EQ(10, *(float *)buf.GetPtr3(1, 2, 0));
	EXPECT_EQ(11, *(float *)buf.GetPtr3(1, 2, 1));
	EXPECT_EQ(12, *(float *)buf.GetPtr3(2, 0, 0));
	EXPECT_EQ(13, *(float *)buf.GetPtr3(2, 0, 1));
	EXPECT_EQ(14, *(float *)buf.GetPtr3(2, 1, 0));
	EXPECT_EQ(15, *(float *)buf.GetPtr3(2, 1, 1));
	EXPECT_EQ(16, *(float *)buf.GetPtr3(2, 2, 0));
	EXPECT_EQ(17, *(float *)buf.GetPtr3(2, 2, 1));
	EXPECT_EQ(18, *(float *)buf.GetPtr3(3, 0, 0));
	EXPECT_EQ(19, *(float *)buf.GetPtr3(3, 0, 1));
	EXPECT_EQ(20, *(float *)buf.GetPtr3(3, 1, 0));
	EXPECT_EQ(21, *(float *)buf.GetPtr3(3, 1, 1));
	EXPECT_EQ(22, *(float *)buf.GetPtr3(3, 2, 0));
	EXPECT_EQ(23, *(float *)buf.GetPtr3(3, 2, 1));

	// シーケンシャルアクセス確認
	{
		int i = 0;
		buf.ResetPtr();
		while (!buf.IsEnd()) {
			EXPECT_EQ((float)i, *(float *)buf.NextPtr());
			i++;
		}
		EXPECT_EQ(i, 24);
	}

#if BB_NEURALNET_BUFFER_USE_ROI
	// オフセットのみのROI
	buf.SetRoi({ 0, 1, 0 });
	EXPECT_EQ(2, *(float *)buf.GetPtr3(0, 0, 0));
	EXPECT_EQ(3, *(float *)buf.GetPtr3(0, 0, 1));
	EXPECT_EQ(4, *(float *)buf.GetPtr3(0, 1, 0));
	EXPECT_EQ(5, *(float *)buf.GetPtr3(0, 1, 1));
	EXPECT_EQ(8, *(float *)buf.GetPtr3(1, 0, 0));
	EXPECT_EQ(9, *(float *)buf.GetPtr3(1, 0, 1));
	EXPECT_EQ(10, *(float *)buf.GetPtr3(1, 1, 0));
	EXPECT_EQ(11, *(float *)buf.GetPtr3(1, 1, 1));
	EXPECT_EQ(14, *(float *)buf.GetPtr3(2, 0, 0));
	EXPECT_EQ(15, *(float *)buf.GetPtr3(2, 0, 1));
	EXPECT_EQ(16, *(float *)buf.GetPtr3(2, 1, 0));
	EXPECT_EQ(17, *(float *)buf.GetPtr3(2, 1, 1));
	EXPECT_EQ(20, *(float *)buf.GetPtr3(3, 0, 0));
	EXPECT_EQ(21, *(float *)buf.GetPtr3(3, 0, 1));
	EXPECT_EQ(22, *(float *)buf.GetPtr3(3, 1, 0));
	EXPECT_EQ(23, *(float *)buf.GetPtr3(3, 1, 1));

	buf.ResetPtr();
	EXPECT_EQ(false, buf.IsEnd());
	EXPECT_EQ(2, *(float *)buf.NextPtr());
	EXPECT_EQ(false, buf.IsEnd());
	EXPECT_EQ(3, *(float *)buf.NextPtr());
	EXPECT_EQ(4, *(float *)buf.NextPtr());
	EXPECT_EQ(5, *(float *)buf.NextPtr());
	EXPECT_EQ(8, *(float *)buf.NextPtr());
	EXPECT_EQ(9, *(float *)buf.NextPtr());
	EXPECT_EQ(10, *(float *)buf.NextPtr());
	EXPECT_EQ(11, *(float *)buf.NextPtr());
	EXPECT_EQ(14, *(float *)buf.NextPtr());
	EXPECT_EQ(15, *(float *)buf.NextPtr());
	EXPECT_EQ(16, *(float *)buf.NextPtr());
	EXPECT_EQ(17, *(float *)buf.NextPtr());
	EXPECT_EQ(20, *(float *)buf.NextPtr());
	EXPECT_EQ(21, *(float *)buf.NextPtr());
	EXPECT_EQ(22, *(float *)buf.NextPtr());
	EXPECT_EQ(false, buf.IsEnd());
	EXPECT_EQ(23, *(float *)buf.NextPtr());
	EXPECT_EQ(true, buf.IsEnd());

	buf.SetRoi({ 0, 0, 2 });
	EXPECT_EQ(14, *(float *)buf.GetPtr3(0, 0, 0));
	EXPECT_EQ(15, *(float *)buf.GetPtr3(0, 0, 1));
	EXPECT_EQ(16, *(float *)buf.GetPtr3(0, 1, 0));
	EXPECT_EQ(17, *(float *)buf.GetPtr3(0, 1, 1));
	EXPECT_EQ(20, *(float *)buf.GetPtr3(1, 0, 0));
	EXPECT_EQ(21, *(float *)buf.GetPtr3(1, 0, 1));
	EXPECT_EQ(22, *(float *)buf.GetPtr3(1, 1, 0));
	EXPECT_EQ(23, *(float *)buf.GetPtr3(1, 1, 1));

	EXPECT_EQ(14, *(float *)buf.GetPtr(0));
	EXPECT_EQ(15, *(float *)buf.GetPtr(1));
	EXPECT_EQ(16, *(float *)buf.GetPtr(2));
	EXPECT_EQ(17, *(float *)buf.GetPtr(3));
	EXPECT_EQ(20, *(float *)buf.GetPtr(4));
	EXPECT_EQ(21, *(float *)buf.GetPtr(5));
	EXPECT_EQ(22, *(float *)buf.GetPtr(6));
	EXPECT_EQ(23, *(float *)buf.GetPtr(7));

	buf.ResetPtr();
	EXPECT_EQ(false, buf.IsEnd());
	EXPECT_EQ(14, *(float *)buf.NextPtr());
	EXPECT_EQ(false, buf.IsEnd());
	EXPECT_EQ(15, *(float *)buf.NextPtr());
	EXPECT_EQ(16, *(float *)buf.NextPtr());
	EXPECT_EQ(17, *(float *)buf.NextPtr());
	EXPECT_EQ(20, *(float *)buf.NextPtr());
	EXPECT_EQ(21, *(float *)buf.NextPtr());
	EXPECT_EQ(false, buf.IsEnd());
	EXPECT_EQ(22, *(float *)buf.NextPtr());
	EXPECT_EQ(false, buf.IsEnd());
	EXPECT_EQ(23, *(float *)buf.NextPtr());
	EXPECT_EQ(true, buf.IsEnd());


	// ROI解除
	buf.ClearRoi();
#endif

	// シーケンシャルアクセス確認
	{
		int i = 0;
		buf.ResetPtr();
		while (!buf.IsEnd()) {
			EXPECT_EQ((float)i, *(float *)buf.NextPtr());
			i++;
		}
		EXPECT_EQ(i, 24);
	}

#if	BB_NEURALNET_BUFFER_USE_ROI
	// 範囲付きROI
	buf.SetRoi({ 0, 1, 1 }, { 1, 2, 2 });

	EXPECT_EQ(8, *(float *)buf.GetPtr(0));	// (1, 1, 0) : 8
	EXPECT_EQ(10, *(float *)buf.GetPtr(1));	// (1, 2, 0) : 8
	EXPECT_EQ(14, *(float *)buf.GetPtr(2));	// (2, 1, 0) : 14
	EXPECT_EQ(16, *(float *)buf.GetPtr(3));	// (2, 2, 0) : 16

	buf.ResetPtr();
	EXPECT_EQ(false, buf.IsEnd());
	EXPECT_EQ(8, *(float *)buf.NextPtr());
	EXPECT_EQ(false, buf.IsEnd());
	EXPECT_EQ(10, *(float *)buf.NextPtr());
	EXPECT_EQ(false, buf.IsEnd());
	EXPECT_EQ(14, *(float *)buf.NextPtr());
	EXPECT_EQ(false, buf.IsEnd());
	EXPECT_EQ(16, *(float *)buf.NextPtr());
	EXPECT_EQ(true, buf.IsEnd());
#endif
}


#if	BB_NEURALNET_BUFFER_USE_ROI

TEST(NeuralNetBufferTest, testNeuralNetBufferTest2)
{
	bb::NeuralNetBuffer<> base_buf(2, 2 * 3 * 4, BB_TYPE_REAL32);
	
	// 入力データ作成
	for (size_t node = 0; node < 24; node++) {
		base_buf.SetReal(0, node, (float)node);
		base_buf.SetReal(1, node, (float)node + 1000);
	}
	
	auto buf = base_buf;
	buf.SetDimensions({ 4, 3, 2 });
	
	//  0  1  2  3
	//  4  5  6  7
	//  8  9 10 11
	//
	// 12 13 14 15
	// 16 17 18 19
	// 20 21 22 23

	
	buf.SetRoi({ 0, 0, 0 }, { 2, 2, 2 });

	EXPECT_EQ(0, ((float*)buf.GetPtr(0))[0]);
	EXPECT_EQ(1, ((float*)buf.GetPtr(1))[0]);
	EXPECT_EQ(4, ((float*)buf.GetPtr(2))[0]);
	EXPECT_EQ(5, ((float*)buf.GetPtr(3))[0]);
	EXPECT_EQ(12, ((float*)buf.GetPtr(4))[0]);
	EXPECT_EQ(13, ((float*)buf.GetPtr(5))[0]);
	EXPECT_EQ(16, ((float*)buf.GetPtr(6))[0]);
	EXPECT_EQ(17, ((float*)buf.GetPtr(7))[0]);

	EXPECT_EQ(0, buf.GetReal(0, 0));
	EXPECT_EQ(1, buf.GetReal(0, 1));
	EXPECT_EQ(4, buf.GetReal(0, 2));
	EXPECT_EQ(5, buf.GetReal(0, 3));
	EXPECT_EQ(12, buf.GetReal(0, 4));
	EXPECT_EQ(13, buf.GetReal(0, 5));
	EXPECT_EQ(16, buf.GetReal(0, 6));
	EXPECT_EQ(17, buf.GetReal(0, 7));

	EXPECT_EQ(0+1000, buf.GetReal(1, 0));
	EXPECT_EQ(1+1000, buf.GetReal(1, 1));
	EXPECT_EQ(4+1000, buf.GetReal(1, 2));
	EXPECT_EQ(5+1000, buf.GetReal(1, 3));
	EXPECT_EQ(12+1000, buf.GetReal(1, 4));
	EXPECT_EQ(13+1000, buf.GetReal(1, 5));
	EXPECT_EQ(16+1000, buf.GetReal(1, 6));
	EXPECT_EQ(17+1000, buf.GetReal(1, 7));

	buf.ClearRoi();
	buf.SetRoi({ 1, 0, 0 }, { 2, 2, 2 });
	EXPECT_EQ(1, buf.GetReal(0, 0));
	EXPECT_EQ(2, buf.GetReal(0, 1));
	EXPECT_EQ(5, buf.GetReal(0, 2));
	EXPECT_EQ(6, buf.GetReal(0, 3));
	EXPECT_EQ(13, buf.GetReal(0, 4));
	EXPECT_EQ(14, buf.GetReal(0, 5));
	EXPECT_EQ(17, buf.GetReal(0, 6));
	EXPECT_EQ(18, buf.GetReal(0, 7));

	buf.ClearRoi();
	buf.SetRoi({ 1, 1, 0 }, { 2, 2, 2 });
	EXPECT_EQ(5, buf.GetReal(0, 0));
	EXPECT_EQ(6, buf.GetReal(0, 1));
	EXPECT_EQ(9, buf.GetReal(0, 2));
	EXPECT_EQ(10, buf.GetReal(0, 3));
	EXPECT_EQ(17, buf.GetReal(0, 4));
	EXPECT_EQ(18, buf.GetReal(0, 5));
	EXPECT_EQ(21, buf.GetReal(0, 6));
	EXPECT_EQ(22, buf.GetReal(0, 7));

	buf.ClearRoi();
	buf.SetRoi({ 1, 1, 1 }, { 2, 2, 1 });
	EXPECT_EQ(17, buf.GetReal(0, 0));
	EXPECT_EQ(18, buf.GetReal(0, 1));
	EXPECT_EQ(21, buf.GetReal(0, 2));
	EXPECT_EQ(22, buf.GetReal(0, 3));


}


TEST(NeuralNetBufferTest, testNeuralNetBufferTest3)
{
	size_t input_c_size = 1;
	size_t input_h_size = 15;
	size_t input_w_size = 14;
	size_t output_c_size = 1;
	size_t output_h_size = 9;
	size_t output_w_size = 8;
	size_t y_step = 1;
	size_t x_step = 1;
	size_t filter_h_size = 5;
	size_t filter_w_size = 5;

	size_t input_node_size = input_c_size * input_h_size * input_w_size;
	size_t output_node_size = output_c_size * output_h_size * output_w_size;

	bb::NeuralNetBuffer<> in_buf(1, input_node_size, BB_TYPE_BINARY);
	bb::NeuralNetBuffer<> out_buf(1, output_node_size, BB_TYPE_BINARY);

	// 入力データ作成
	std::mt19937_64 mt(1);
	for (size_t node = 0; node < input_node_size; node++) {
		in_buf.SetBinary(0, node, mt() % 2 != 0);
	}

	auto in_val = in_buf;
	auto out_val = out_buf;
	in_val.SetDimensions({ input_w_size, input_h_size, input_c_size });
	out_val.SetDimensions({ output_w_size, output_h_size, output_c_size });

	size_t in_y = 0;
	for (size_t out_y = 0; out_y < output_h_size; out_y++) {
		size_t in_x = 0;
		for (size_t out_x = 0; out_x < output_w_size; out_x++) {
			in_val.ClearRoi();
			in_val.SetRoi({ in_x, in_y, 0}, { filter_w_size , filter_h_size , input_c_size });
			out_val.ClearRoi();
			out_val.SetRoi({ out_x, out_y, 0}, { 1, 1, output_c_size });

			out_val.SetBinary(0, 0, in_val.GetBinary(0, 0));
			in_x += x_step;
		}
		in_y += y_step;
	}

	for (size_t y = 0; y < output_h_size; y++) {
		for (size_t x = 0; x < output_w_size; x++) {
			EXPECT_EQ(in_buf.GetBinary(0, y*input_w_size+x), out_buf.GetBinary(0, y*output_w_size + x));
		}
	}
}

#endif

#endif
