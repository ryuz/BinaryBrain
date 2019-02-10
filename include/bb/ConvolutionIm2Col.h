// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once

#include <fstream>
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
	index_t 		m_input_frame_size;
	index_t			m_output_frame_size;
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
	inline index_t GetInputNode(index_t c, index_t y, index_t x)
	{
		return (c * m_input_h_size + y)*m_input_w_size + x;
	}

	inline index_t GetOutputNode(index_t c, index_t y, index_t x)
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
        m_input_frame_size = x.GetFrameSize();
        m_output_frame_size = m_input_frame_size * m_output_h_size * m_output_w_size;

        m_y.Resize(m_output_frame_size, m_output_shape, x.GetType());

#ifdef BB_WITH_CUDA
        if ( type == BB_TYPE_FP32 && x.IsDeviceAvailable() && m_y.IsDeviceAvailable())
        {
            auto ptr_x = x.GetDevConstPtr();
            auto ptr_y = m_y.GetDevPtr();
            cubb_Im2Col_Forward(
                (const float *)ptr_x.GetAddr(),
                (int)m_input_frame_size,
                (int)x.GetFrameStride() / sizeof(float),
                (int)m_input_w_size,
                (int)m_input_h_size,
                (int)m_input_c_size,
                (float *)ptr_y.GetAddr(),
                (int)m_y.GetFrameStride() / sizeof(float),
                (int)m_filter_w_size,
                (int)m_filter_h_size);
            return m_y;
        }
#endif


   		const index_t frame_size = m_y.GetFrameStride() * 8 / DataType<ST>::bit_size;
		const index_t frame_unit = 256 / DataType<ST>::bit_size;

        auto ptr_x = x.GetConstPtr();
        auto ptr_y = m_y.GetPtr();
        auto addr_x = ptr_x.GetAddr();
        auto addr_y = ptr_y.GetAddr();

   		for (index_t c = 0; c < m_input_c_size; ++c) {
#pragma omp parallel for
			for (index_t frame_base = 0; frame_base < frame_size; frame_base += frame_unit) {
				for (index_t fy = 0; fy < m_filter_h_size; ++fy) {
					for (index_t fx = 0; fx < m_filter_w_size; ++fx) {
						index_t output_node = GetOutputNode(c, fy, fx);
						for (index_t frame_step = 0; frame_step < frame_unit; ++frame_step) {
							index_t output_frame = frame_base + frame_step;
							index_t input_frame = output_frame / (m_output_h_size * m_output_w_size);
							index_t f = output_frame % (m_output_h_size * m_output_w_size);
							index_t ix = f % m_output_w_size;
							index_t iy = f / m_output_w_size;
							ix += fx;
							iy += fy;
    						index_t input_node = GetInputNode(c, iy, ix);
							ST sig = x.template Get<ST, ST>(addr_x, input_frame, input_node);
							m_y.template Set<ST, ST>(addr_y, output_frame, output_node, sig);
						}
					}
				}
			}
		}

        return m_y;
    }


	FrameBuffer Backward(const FrameBuffer& dy)
	{
        BB_ASSERT(dy.GetType() == DataType<GT>::type);
        m_dx.Resize(m_input_frame_size, m_input_shape, DataType<GT>::type);

#ifdef BB_WITH_CUDA
        if ( dy.GetType() == BB_TYPE_FP32 && dy.IsDeviceAvailable() && m_dx.IsDeviceAvailable())
        {
            auto ptr_dy = dy.GetDevConstPtr();
            auto ptr_dx = m_dx.GetDevPtr();
            cubb_Im2Col_Backward(
                (float *)ptr_dx.GetAddr(),
                (int)m_input_frame_size,
                (int)m_dx.GetFrameStride() / sizeof(float),
                (int)m_input_w_size,
                (int)m_input_h_size,
                (int)m_input_c_size,
                (const float *)ptr_dy.GetAddr(),
                (int)dy.GetFrameStride() / sizeof(float),
                (int)m_filter_w_size,
                (int)m_filter_h_size);
            return m_dx;
        }
#endif


		const index_t frame_size = dy.GetFrameStride() * 8 / DataType<GT>::bit_size;
		const index_t frame_unit = 256 / DataType<GT>::bit_size;

        auto ptr_dy = dy.GetConstPtr();
        auto ptr_dx = m_dx.GetPtr();
        auto addr_dy = ptr_dy.GetAddr();
        auto addr_dx = ptr_dx.GetAddr();

   		m_dx.FillZero();

		for (index_t c = 0; c < m_input_c_size; ++c) {
#pragma omp parallel for
			for (index_t frame_base = 0; frame_base < frame_size; frame_base += frame_unit) {
				for (index_t fy = 0; fy < m_filter_h_size; ++fy) {
					for (index_t fx = 0; fx < m_filter_w_size; ++fx) {
						index_t output_node = GetOutputNode(c, fy, fx);
						for (index_t frame_step = 0; frame_step < frame_unit; ++frame_step) {
							index_t output_frame = frame_base + frame_step;
							index_t input_frame = output_frame / (m_output_h_size * m_output_w_size);
							index_t f = output_frame % (m_output_h_size * m_output_w_size);
							index_t ix = f % m_output_w_size;
							index_t iy = f / m_output_w_size;
							ix += fx;
							iy += fy;
							index_t input_node = GetInputNode(c, iy, ix);
							GT grad = dy.template Get<GT, GT>(addr_dy, output_frame, output_node);
							m_dx.template Add<ST, ST>(addr_dx, input_frame, input_node, grad);
						}
					}
				}
			}
		}

        return m_dx;
	}
};


}