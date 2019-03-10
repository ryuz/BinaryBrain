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

#include "bb/Manager.h"
#include "bb/Model.h"
#include "bb/FrameBuffer.h"


namespace bb {


template <typename FT = float, typename BT = float>
class ConvolutionIm2Col : public Model
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

protected:
	ConvolutionIm2Col() {}

public:
	~ConvolutionIm2Col() {}

    struct create_t
    {
        index_t filter_h_size = 3;
        index_t filter_w_size = 3;
    };

	static std::shared_ptr<ConvolutionIm2Col> Create(create_t const & create)
	{
        auto self = std::shared_ptr<ConvolutionIm2Col>(new ConvolutionIm2Col);
		self->m_filter_h_size = create.filter_h_size;
        self->m_filter_w_size = create.filter_w_size;
        return self;
	}

	static std::shared_ptr<ConvolutionIm2Col> Create(size_t filter_h_size, size_t filter_w_size)
    {
        auto self = std::shared_ptr<ConvolutionIm2Col>(new ConvolutionIm2Col);
		self->m_filter_h_size = filter_h_size;
        self->m_filter_w_size = filter_w_size;
        return self;
    }

	std::string GetClassName(void) const { return "ConvolutionIm2Col"; }


    /**
     * @brief  入力のshape設定
     * @detail 入力のshape設定
     * @param shape 新しいshape
     * @return なし
     */
    indices_t SetInputShape(indices_t shape)
    {
        // 形状設定
        m_input_shape = shape;
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

        return m_output_shape;
    }
    
   /**
     * @brief  入力形状取得
     * @detail 入力形状を取得する
     * @return 入力形状を返す
     */
    indices_t GetInputShape(void) const
    {
        return m_input_shape;
    }

    /**
     * @brief  出力形状取得
     * @detail 出力形状を取得する
     * @return 出力形状を返す
     */
    indices_t GetOutputShape(void) const
    {
        return m_output_shape;
    }


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

    FrameBuffer Forward(FrameBuffer x, bool train = true)
    {
        BB_ASSERT(x.GetType() == DataType<FT>::type);

        // SetInputShpaeされていなければ初回に設定
        if ( x.GetShape() != m_input_shape ) {
            SetInputShape(x.GetShape());
        }

        /*
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
        */

        m_input_frame_size = x.GetFrameSize();
        m_output_frame_size = m_input_frame_size * m_output_h_size * m_output_w_size;

        m_y.Resize(x.GetType(), m_output_frame_size, m_output_shape);

#ifdef BB_WITH_CUDA
        if ( x.GetType() == BB_TYPE_FP32 && x.IsDeviceAvailable() && m_y.IsDeviceAvailable() && Manager::IsDeviceAvailable())
        {
            auto ptr_x = x.LockDeviceMemoryConst();
            auto ptr_y = m_y.LockDeviceMemory();
            cubb_fp32_Im2Col_Forward(
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


   		const index_t frame_size = m_y.GetFrameStride() * 8 / DataType<FT>::bit_size;
		const index_t frame_unit = 256 / DataType<FT>::bit_size;

        auto ptr_x = x.LockMemoryConst();
        auto ptr_y = m_y.LockMemory();
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
							FT sig = x.template Get<FT, FT>(addr_x, input_frame, input_node);
							m_y.template Set<FT, FT>(addr_y, output_frame, output_node, sig);
						}
					}
				}
			}
		}

        return m_y;
    }


	FrameBuffer Backward(FrameBuffer dy)
	{
        BB_ASSERT(dy.GetType() == DataType<BT>::type);
        m_dx.Resize(DataType<BT>::type, m_input_frame_size, m_input_shape);

#ifdef BB_WITH_CUDA
        if ( dy.GetType() == BB_TYPE_FP32 && dy.IsDeviceAvailable() && m_dx.IsDeviceAvailable() && Manager::IsDeviceAvailable())
        {
            auto ptr_dy = dy.LockDeviceMemoryConst();
            auto ptr_dx = m_dx.LockDeviceMemory();
            cubb_fp32_Im2Col_Backward(
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


		const index_t frame_size = dy.GetFrameStride() * 8 / DataType<BT>::bit_size;
		const index_t frame_unit = 256 / DataType<BT>::bit_size;

        auto ptr_dy = dy.LockMemoryConst();
        auto ptr_dx = m_dx.LockMemory();
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
							BT grad = dy.template Get<BT, BT>(addr_dy, output_frame, output_node);
							m_dx.template Add<BT, BT>(addr_dx, input_frame, input_node, grad);
						}
					}
				}
			}
		}

        return m_dx;
	}
};


}