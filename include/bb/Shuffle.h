// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once


#include "bb/Manager.h"
#include "bb/Model.h"


namespace bb {


// Shuffle
class Shuffle : public Model
{
    using _super = Model;

public:
    static inline std::string ModelName(void) { return "Shuffle"; }
    static inline std::string ObjectName(void){ return ModelName(); }

    std::string GetModelName(void)  const override { return ModelName(); }
    std::string GetObjectName(void) const override { return ObjectName(); }

protected:
    bool        m_host_only = false;

    indices_t   m_input_shape;
    indices_t   m_output_shape;
    index_t     m_shuffle_unit = 0;

public:
    // 生成情報
    struct create_t
    {
        index_t     shuffle_unit = 0;
        indices_t   output_shape;
    };
    
protected:
    Shuffle() {}

    Shuffle(create_t const &create)
    {
        m_output_shape = create.output_shape;
        m_shuffle_unit = create.shuffle_unit;
    }
    
    /**
     * @brief  コマンド処理
     * @detail コマンド処理
     * @param  args   コマンド
     */
    void CommandProc(std::vector<std::string> args)
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
        os << indent << " shuffle_unit : " << m_shuffle_unit << std::endl;
    }

public:
    ~Shuffle() {}

    static std::shared_ptr<Shuffle> Create(create_t const &create)
    {
        return std::shared_ptr<Shuffle>(new Shuffle(create));
    }

    static std::shared_ptr<Shuffle> Create(index_t shuffle_unit, indices_t output_shape=indices_t())
    {
        create_t create;
        create.shuffle_unit = shuffle_unit;
        create.output_shape = output_shape;
        return Create(create);
    }

    static std::shared_ptr<Shuffle> Create(index_t shuffle_unit, index_t output_node_size)
    {
        return Create(shuffle_unit, indices_t({output_node_size}));
    }

    static std::shared_ptr<Shuffle> Create(void)
    {
        return Create(create_t());
    }

#ifdef BB_PYBIND11
    static std::shared_ptr<Shuffle> CreatePy(index_t shuffle_unit, indices_t output_shape=indices_t())
    {
        create_t create;
        create.shuffle_unit = shuffle_unit;
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
    indices_t SetInputShape(indices_t shape) override
    {
        m_input_shape = shape;

        if ( m_output_shape.empty() || CalcShapeSize(shape) != CalcShapeSize(m_output_shape) ) {
            m_output_shape = m_input_shape;
        }

        BB_ASSERT(CalcShapeSize(m_output_shape) % m_shuffle_unit == 0);

        return m_output_shape;
    }

    /**
     * @brief  入力形状取得
     * @detail 入力形状を取得する
     * @return 入力形状を返す
     */
    indices_t GetInputShape(void) const override
    {
        return m_input_shape;
    }

    /**
     * @brief  出力形状取得
     * @detail 出力形状を取得する
     * @return 出力形状を返す
     */
    indices_t GetOutputShape(void) const override
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
    inline FrameBuffer Forward(FrameBuffer x_buf, bool train = true) override
    {
        // 戻り値のサイズ設定
        FrameBuffer y_buf( x_buf.GetFrameSize(), m_output_shape, x_buf.GetType());

#ifdef BB_WITH_CUDA
        if ( !m_host_only && x_buf.IsDeviceAvailable() && y_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
            // CUDA版
            auto ptr_x = x_buf.LockDeviceMemoryConst();
            auto ptr_y = y_buf.LockDeviceMemory(true);
            bbcu_Shuffle_Forward<int>(
                        (int const   *)ptr_x.GetAddr(),
                        (int         *)ptr_y.GetAddr(),
                        (unsigned int )m_shuffle_unit,
                        (unsigned int )x_buf.GetNodeSize(),
                        (unsigned int )x_buf.GetFrameSize(),
                        (unsigned int )(x_buf.GetFrameStride() / sizeof(int))
                    );
            return y_buf;
        }
#endif

        {
            // 汎用版
            index_t frame_size  = x_buf.GetFrameSize();
            index_t node_size   = x_buf.GetNodeSize();
            index_t stride_size = x_buf.GetFrameStride();

            index_t y_unit_size  = m_shuffle_unit;
            index_t x_unit_size  = node_size / y_unit_size;

            auto x_ptr = (std::uint8_t *)x_buf.LockMemoryConst().GetAddr();
            auto y_ptr = (std::uint8_t *)y_buf.LockMemory().GetAddr();

            #pragma omp parallel for
            for ( index_t i = 0; i < x_unit_size; ++i ) {
                for ( index_t j = 0; j < y_unit_size; ++j ) {
                    memcpy(&y_ptr[(i*y_unit_size+j)*stride_size], &x_ptr[(j*x_unit_size+i)*stride_size], stride_size);
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
    inline FrameBuffer Backward(FrameBuffer dy_buf) override
    {
        // 戻り値のサイズ設定
        FrameBuffer dx_buf(dy_buf.GetFrameSize(), m_input_shape, dy_buf.GetType());
        

        #ifdef BB_WITH_CUDA
        if ( !m_host_only && dy_buf.IsDeviceAvailable() && dx_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
            // CUDA版
            auto ptr_dy = dy_buf.LockDeviceMemoryConst();
            auto ptr_dx = dx_buf.LockDeviceMemory(true);
            bbcu_Shuffle_Backward<int>(
                        (int const   *)ptr_dy.GetAddr(),
                        (int         *)ptr_dx.GetAddr(),
                        (unsigned int )m_shuffle_unit,
                        (unsigned int )dy_buf.GetNodeSize(),
                        (unsigned int )dy_buf.GetFrameSize(),
                        (unsigned int )(dy_buf.GetFrameStride() / sizeof(int))
                    );
            return dx_buf;
        }
#endif

        {
            // 汎用版
            index_t frame_size  = dy_buf.GetFrameSize();
            index_t node_size   = dy_buf.GetNodeSize();
            index_t stride_size = dy_buf.GetFrameStride();

            index_t y_unit_size = m_shuffle_unit;
            index_t x_unit_size = node_size / y_unit_size;

            auto dy_ptr = (std::uint8_t *)dy_buf.LockMemoryConst().GetAddr();
            auto dx_ptr = (std::uint8_t *)dx_buf.LockMemory().GetAddr();

            #pragma omp parallel for
            for ( index_t i = 0; i < y_unit_size; ++i ) {
                for ( index_t j = 0; j < x_unit_size; ++j ) {
                    memcpy(&dx_ptr[(i*x_unit_size+j)*stride_size], &dy_ptr[(j*y_unit_size+i)*stride_size], stride_size);
                }
            }

            return dx_buf;
        }
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
        bb::SaveValue(os, m_host_only);
        bb::SaveValue(os, m_input_shape);
        bb::SaveValue(os, m_output_shape);
        bb::SaveValue(os, m_shuffle_unit);
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
        bb::LoadValue(is, m_host_only);
        bb::LoadValue(is, m_input_shape);
        bb::LoadValue(is, m_output_shape);
        bb::LoadValue(is, m_shuffle_unit);
                
        // 再構築
    }
};


}


// end of file
