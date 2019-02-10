// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once


#include <vector>
#include <random>

#include "bb/Layer.h"
#include "bb/FrameBuffer.h"


namespace bb {


template <typename ST = float, typename GT = float>
class ConvolutionIm2Col //  : public Layer
{
protected:
    indices_t       m_input_shape;
    indices_t       m_output_shape;
	index_t 		m_input_frame_size = 1;
	index_t			m_output_frame_size = 1;
	index_t			m_input_h_size;
	index_t			m_input_w_size;
	index_t			m_input_c_size;
	index_t			m_filter_h_size;
	index_t			m_filter_w_size;
	index_t			m_output_h_size;
	index_t			m_output_w_size;

    // メモリの確保/開放を繰り返さないように演算後も確保
    FrameBuffer     m_y;
    FrameBuffer     m_dx;

public:
	ConvolutionIm2Col() {}
	
    struct construct_t
    {
        index_t filter_h_size = 3;
        index_t filter_w_size = 3;
    };

	ConvolutionIm2Col(construct_t const & construct)
	{
		m_filter_h_size = construct.filter_h_size;
        m_filter_w_size = construct.filter_w_size;
	}

    ConvolutionIm2Col(size_t filter_h_size, size_t filter_w_size)
    {
		m_filter_h_size = filter_h_size;
        m_filter_w_size = filter_w_size;
    }


	~ConvolutionIm2Col() {}

	std::string GetClassName(void) const { return "ConvolutionIm2Col"; }



protected:
	inline int GetInputNode(int c, int y, int x)
	{
		return (c * m_input_h_size + y)*m_input_w_size + x;
	}

	inline int GetOutputNode(int c, int y, int x)
	{
		return (c*m_filter_h_size + y)*m_filter_w_size + x;
	}

public:
    FrameBuffer Forward(FrameBuffer const &x, bool train = true)
    {
        m_input_shape = x.GetShape();
        BB_ASSERT(m_input_shape.size() == 3);

        m_input_w_size = m_input_shape[0];
        m_input_h_size = m_input_shape[1];
        m_input_c_size = m_input_shape[2];
		m_output_h_size = m_input_h_size - m_filter_h_size + 1;
		m_output_w_size = m_input_w_size - m_filter_w_size + 1;

        m_output_shape.resize(3);
        m_output_shape[0] = m_filter_w_size;
        m_output_shape[1] = m_filter_h_size;
        m_output_shape[2] = m_input_c_size;

        auto type = x.GetType();
        BB_ASSERT(type == DataType<ST>::type);
        auto input_frame_size = x.GetFrameSize();
        auto output_frame_size = input_frame_size * m_output_h_size * m_output_w_size;

        m_y.Resize(output_frame_size, m_output_shape, x.GetType());

#ifdef BB_WITH_CUDA
        if ( type == BB_TYPE_FP32 && x.IsDeviceAvailable() && m_y.IsDeviceAvailable())
        {
            auto ptr_x = x.GetDevConstPtr();
            auto ptr_y = m_y.GetDevPtr();
            cubb_Im2Col_Forward(
                (const float *)ptr_x.GetAddr(),
                (float *)ptr_y.GetAddr(),
                (int)input_frame_size,
                (int)m_input_w_size,
                (int)m_input_h_size,
                (int)m_input_c_size,
                (int)m_filter_w_size,
                (int)m_filter_h_size);
            return m_y;
        }
#endif

   		const int frame_size = m_y.GetFrameStride() * 8 / DataType<ST>::bit_size;
		const int frame_unit = 256 / DataType<ST>::bit_size;

        auto ptr_x = x.GetConstPtr();
        auto ptr_y = m_y.GetPtr();
        auto addr_x = ptr_x.GetAddr();
        auto addr_y = ptr_y.GetAddr();

   		for (int c = 0; c < m_input_c_size; ++c) {
#pragma omp parallel for
			for (int frame_base = 0; frame_base < frame_size; frame_base += frame_unit) {
				for (int fy = 0; fy < m_filter_h_size; ++fy) {
					for (int fx = 0; fx < m_filter_w_size; ++fx) {
						int output_node = GetOutputNode(c, fy, fx);
						for (int frame_step = 0; frame_step < frame_unit; ++frame_step) {
							int output_frame = frame_base + frame_step;
							int input_frame = output_frame / (m_output_h_size * m_output_w_size);
							int f = output_frame % (m_output_h_size * m_output_w_size);
							int ix = f % m_output_w_size;
							int iy = f / m_output_w_size;
							ix += fx;
							iy += fy;
    						int input_node = GetInputNode(c, iy, ix);
							ST sig = x.template Get<ST, ST>(addr_x, input_frame, input_node);
							m_y.template Set<ST, ST>(addr_y, output_frame, output_node, sig);

//							ST sig = x.template Get<ST, ST>(input_frame, input_node);
//							m_y.template Set<ST, ST>(output_frame, output_node, sig);
						}
					}
				}
			}
		}

        return m_y;
    }


	FrameBuffer Backward(FrameBuffer dy)
	{
		m_dx.FillZero();

		const index_t frame_size = dy.GetFrameStride() * 8 / DataType<GT>::bit_size;
		const index_t frame_unit = 256 / NeuralNetType<GT>::bit_size;

		for (int c = 0; c < m_input_c_size; ++c) {
#pragma omp parallel for
			for (int frame_base = 0; frame_base < frame_size; frame_base += frame_unit) {
				for (int fy = 0; fy < m_filter_h_size; ++fy) {
					for (int fx = 0; fx < m_filter_w_size; ++fx) {
						int output_node = GetOutputNode(c, fy, fx);
						for (int frame_step = 0; frame_step < frame_unit; ++frame_step) {
							int output_frame = frame_base + frame_step;
							int input_frame = output_frame / (m_output_h_size * m_output_w_size);
							int f = output_frame % (m_output_h_size * m_output_w_size);
							int ix = f % m_output_w_size;
							int iy = f / m_output_w_size;
							ix += fx;
							iy += fy;
							int input_node = GetInputNode(c, iy, ix);
							GT err = out_err_buf.template Get<ET>(output_frame, output_node);
							in_err_buf.template Set<ET>(input_frame, input_node, in_err_buf.template Get<ET>(input_frame, input_node) + err);
						}
					}
				}
			}
		}
	}
};


}