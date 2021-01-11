// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                 Copyright (C) 2018-2020 by Ryuji Fuchikami
//                                 https://github.com/ryuz
//                                 ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once


#include "bb/Manager.h"
#include "bb/Model.h"


namespace bb {


template<typename BinType=float, typename RealType=float>
class BitEncode : public Model
{
    using _super = Model;

public:
    static inline std::string ModelName(void) { return "BitEncode"; }
    static inline std::string ObjectName(void){ return ModelName() + "_" + DataType<BinType>::Name() + "_" + DataType<RealType>::Name(); }

    std::string GetModelName(void)  const override { return ModelName(); }
    std::string GetObjectName(void) const override { return ObjectName(); }

protected:
    bool        m_host_only = false;
    
    index_t     m_bit_size = 0;
    indices_t   m_input_shape;
    indices_t   m_output_shape;

public:
    // 生成情報
    struct create_t
    {
        index_t     bit_size = 1;
        indices_t   output_shape;
    };
    
protected:
    BitEncode() {}

    BitEncode(create_t const &create)
    {
        m_bit_size     = create.bit_size;
        m_output_shape = create.output_shape;
    }
    
    /**
     * @brief  コマンド処理
     * @detail コマンド処理
     * @param  args   コマンド
     */
    void CommandProc(std::vector<std::string> args) override
    {
        // HostOnlyモード設定
        if (args.size() == 2 && args[0] == "host_only")
        {
            m_host_only = EvalBool(args[1]);
        }
    }

    void PrintInfoText(std::ostream& os, std::string indent, int columns, int nest, int depth) const override
    {
        _super::PrintInfoText(os, indent, columns, nest, depth);
//      os << indent << " input  shape : " << GetInputShape();
//      os << indent << " output shape : " << GetOutputShape();
        os << indent << " bit_size : " << m_bit_size << std::endl;
    }

public:
    ~BitEncode() {}

    static std::shared_ptr<BitEncode> Create(create_t const &create)
    {
        return std::shared_ptr<BitEncode>(new BitEncode(create));
    }

    static std::shared_ptr<BitEncode> Create(index_t bit_size, indices_t output_shape=indices_t())
    {
        create_t create;
        create.bit_size     = bit_size;
        create.output_shape = output_shape;
        return Create(create);
    }

    static std::shared_ptr<BitEncode> Create(void)
    {
        return Create(create_t());
    }

#ifdef BB_PYBIND11
    static std::shared_ptr<BitEncode> CreatePy(index_t bit_size, indices_t output_shape=indices_t())
    {
        create_t create;
        create.bit_size     = bit_size;
        create.output_shape = output_shape;
        return Create(create);
    }
#endif

    /**
     * @brief  入力形状設定
     * @detail 入力形状を設定する
     *         内部変数を初期化し、以降、GetOutputShape()で値取得可能となることとする
     *         同一形状を指定しても内部変数は初期化されるものとする
     * @param  shape      1フレームのノードを構成するshape
     * @return 出力形状を返す
     */
    indices_t SetInputShape(indices_t shape)
    {
        m_input_shape = shape;

        if ( m_output_shape.empty() || CalcShapeSize(shape)*m_bit_size != CalcShapeSize(m_output_shape) ) {
            m_output_shape = m_input_shape;
            m_output_shape[0] *= m_bit_size;
        }

        BB_ASSERT(CalcShapeSize(m_output_shape) % m_bit_size == 0);
        BB_ASSERT(CalcShapeSize(m_output_shape) / m_bit_size == CalcShapeSize(m_input_shape));

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
    
    /**
     * @brief  forward演算
     * @detail forward演算を行う
     * @param  x     入力データ
     * @param  train 学習時にtrueを指定
     * @return forward演算結果
     */
    inline FrameBuffer Forward(FrameBuffer x_buf, bool train = true)
    {
        // 戻り値のサイズ設定
        FrameBuffer y_buf( x_buf.GetFrameSize(), m_output_shape, DataType<BinType>::type);

#ifdef BB_WITH_CUDA
        if ( !m_host_only && DataType<BinType>::type == BB_TYPE_BIT && DataType<RealType>::type == BB_TYPE_FP32 && x_buf.IsDeviceAvailable() && y_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
            // CUDA版
            auto x_ptr = x_buf.LockDeviceMemoryConst();
            auto y_ptr = y_buf.LockDeviceMemory(true);
            bbcu_bit_BitEncode<RealType>(
                        (RealType const *)x_ptr.GetAddr(),
                        (int            *)y_ptr.GetAddr(),
                        (unsigned int    )m_bit_size,
                        (RealType        )0,
                        (RealType        )1,
                        (RealType        )((1 << m_bit_size) - 1),
                        (RealType        )0,
                        (unsigned int    )GetInputNodeSize(),
                        (unsigned int    )x_buf.GetFrameSize(),
                        (unsigned int    )(x_buf.GetFrameStride() / sizeof(RealType)),
                        (unsigned int    )(y_buf.GetFrameStride() / sizeof(int))
                    );
            return y_buf;
        }
#endif

        {
            // 汎用版
            index_t frame_size  = x_buf.GetFrameSize();
            index_t node_size   = x_buf.GetNodeSize();

            auto x_ptr = x_buf.LockConst<RealType>();
            auto y_ptr = y_buf.Lock<BinType>();

            #pragma omp parallel for
            for ( index_t node = 0; node < node_size; ++node ) {
                for ( index_t frame = 0; frame < frame_size; ++frame ) {
                    int x = (int)(x_ptr.Get(frame, node) * ((1 << m_bit_size) - 1));
                    for ( int bit = 0; bit < m_bit_size; ++bit ) {
                        if ( x & (1 << bit) ) {
                            y_ptr.Set(frame, node_size*bit + node, 1);
                        }
                        else {
                            y_ptr.Set(frame, node_size*bit + node, 0);
                        }
                    }
                }
            }

            return y_buf;
        }
    }


   /**
     * @brief  backward演算
     * @detail backward演算を行う
     *         
     * @return backward演算結果
     */
    inline FrameBuffer Backward(FrameBuffer dy_buf)
    {
        // 戻り値のサイズ設定
        FrameBuffer dx_buf(dy_buf.GetFrameSize(), m_input_shape, DataType<RealType>::type);
        dx_buf.FillZero();
        return dx_buf;
    }


    // シリアライズ
protected:
    void DumpObjectData(std::ostream &os) const override
    {
        // バージョン
        std::int64_t ver = 1;
        bb::SaveValue(os, ver);

        // 親クラス
        _super::DumpObjectData(os);

        // メンバ
        bb::SaveValue(os, m_bit_size);
        bb::SaveValue(os, m_input_shape);
        bb::SaveValue(os, m_output_shape);
    }

    void LoadObjectData(std::istream &is) override
    {
        // バージョン
        std::int64_t ver;
        bb::LoadValue(is, ver);

        BB_ASSERT(ver == 1);

        // 親クラス
        _super::LoadObjectData(is);

        // メンバ
        bb::LoadValue(is, m_bit_size);
        bb::LoadValue(is, m_input_shape);
        bb::LoadValue(is, m_output_shape);
        
        // 再構築
        if ( m_output_shape.empty() && !m_input_shape.empty() ) {
            m_output_shape = m_input_shape;
            m_output_shape[0] *= m_bit_size;

            BB_ASSERT(m_bit_size != 0);
            BB_ASSERT(CalcShapeSize(m_output_shape) % m_bit_size == 0);
            BB_ASSERT(CalcShapeSize(m_output_shape) / m_bit_size == CalcShapeSize(m_input_shape));
        }
    }
};


}


// end of file
