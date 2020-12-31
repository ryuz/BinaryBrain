// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                Copyright (C) 2018-2020 by Ryuji Fuchikami
//                                https://github.com/ryuz
//                                ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once


#include <vector>
#include <cmath>


#include "bb/LossFunction.h"


namespace bb {

template <typename BinType = float, typename RealType = float>
class LossBitEncode : public LossFunction
{
protected:
    FrameBuffer m_dy;
    double      m_loss = 0;
    double      m_frames = 0;
    RealType    m_scale;
    RealType    m_offset;

    struct create_t {
        RealType    scale  = (RealType)1.0;
        RealType    offset = (RealType)0.0;
    };

protected:
    LossBitEncode(create_t const &create)
    {
        m_scale  = create.scale;
        m_offset = create.offset;
    }

public:
    ~LossBitEncode() {}

    static std::shared_ptr<LossBitEncode> Create(create_t const &create)
    {
        return std::shared_ptr<LossBitEncode>(new LossBitEncode(create));
    }
    
    static std::shared_ptr<LossBitEncode> Create(RealType scale = (RealType)1.0, RealType offset = (RealType)0.0)
    {
        create_t create;
        create.scale  = scale;
        create.offset = offset;
        return Create(create);
    }

    static std::shared_ptr<LossBitEncode> CreateEx(RealType scale = (RealType)1.0, RealType offset = (RealType)0.0)
    {
        create_t create;
        create.scale  = scale;
        create.offset = offset;
        return Create(create);
    }

    void Clear(void)
    {
        m_loss = 0;
        m_frames = 0;
    }

    double GetLoss(void) const 
    {
        return m_loss / m_frames;
    }

    FrameBuffer CalculateLoss(FrameBuffer y, FrameBuffer t, index_t batch_size)
    {
        index_t frame_size  = y.GetFrameSize();
        index_t y_node_size = y.GetNodeSize();
        index_t t_node_size = t.GetNodeSize();

        BB_ASSERT(frame_size == t.GetFrameSize());
        BB_ASSERT(y_node_size % t_node_size == 0);

        int bit_size = (int)(y_node_size / t_node_size);

        m_dy.Resize(frame_size, {y_node_size}, DataType<RealType>::type);

        auto y_ptr = y.LockConst<BinType>();
        auto t_ptr = t.LockConst<RealType>();
        auto dy_ptr = m_dy.Lock<RealType>();

        for (index_t frame = 0; frame < frame_size; ++frame) {
            for (index_t node = 0; node < t_node_size; ++node) {
                RealType t = (t_ptr.Get(frame, node) - m_offset) * m_scale;
                RealType w = (RealType)1.0;
                for ( int i = 0; i < bit_size; ++i ) {
                    RealType target = (RealType)0.0;
                    RealType y      = y_ptr.Get(frame, node*bit_size+i);
                    if ( t > 0.5 ) {
                        target = (RealType)1.0;
                        t -= 0.5;
                    }
                    auto grad = y - target;
                    dy_ptr.Set(frame, node*bit_size+i, w * grad / (RealType)batch_size);
                    auto error = grad * grad;
                    m_loss += error / (double)t_node_size;

                    t *= (RealType)2.0;
                    w *= (RealType)1.0; // 0.5;
                }
            }
        }
        m_frames += frame_size;

        return m_dy;
    }
};


}

