// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once

#include <array>
#include <vector>
#include "bb/LutLayer.h"


namespace bb {


// テーブルサイズ固定LUT
template <typename T = float>
class StochasticLut2 : public SparseLayer<T, T>
{
    using _super = SparseLayer<T, T>;

protected:
    bool            m_binary_mode = false;
    bool            m_parameter_lock = false;

    index_t         m_input_node_size = 0;
    index_t         m_output_node_size = 0;
    indices_t       m_input_shape;
    indices_t       m_output_shape;

    FrameBuffer     m_x;
    FrameBuffer     m_y;
    FrameBuffer     m_dx;

    Tensor_<std::int32_t>   m_input_index;

    std::shared_ptr<Tensor> m_W;
    std::shared_ptr<Tensor> m_dW;

    std::mt19937_64         m_mt;

protected:
    StochasticLut2() {
        m_W  = std::make_shared<Tensor>();
        m_dW = std::make_shared<Tensor>();
    }

    void CommandProc(std::vector<std::string> args)
    {
        // バイナリモード設定
        if ( args.size() == 2 && args[0] == "binary" )
        {
            m_binary_mode = EvalBool(args[1]);
        }
    }

public:
    ~StochasticLut2() {}

    struct create_t
    {
        indices_t       output_shape;
        std::uint64_t   seed = 1;
    };

    static std::shared_ptr<StochasticLut2> Create(create_t const &create)
    {
        auto self = std::shared_ptr<StochasticLut2>(new StochasticLut2);
        BB_ASSERT(!create.output_shape.empty());
        self->m_output_shape     = create.output_shape;
        self->m_output_node_size = GetShapeSize(self->m_output_shape);
        self->m_mt.seed(create.seed);
        return self;
    }

    static std::shared_ptr<StochasticLut2> Create(indices_t const &output_shape, std::uint64_t seed = 1)
    {
        create_t create;
        create.output_shape = output_shape;
        create.seed         = seed;
        return Create(create);
    }

    static std::shared_ptr<StochasticLut2> Create(index_t output_node_size, std::uint64_t seed = 1)
    {
        create_t create;
        create.output_shape.resize(1);
        create.output_shape[0] = output_node_size;
        return Create(create);
    }

    std::string GetClassName(void) const { return "RealLut2"; }


public:
    void SetParameterLock(bool lock) { m_parameter_lock = lock; }

    // Serialize
    void Save(std::ostream &os) const 
    {
        SaveIndex(os, m_input_node_size);
        SaveIndex(os, m_output_node_size);
        SaveIndices(os, m_input_shape);
        SaveIndices(os, m_output_shape);
        m_input_index.Save(os);
        m_W->Save(os);
    }

    void Load(std::istream &is)
    {
        m_input_node_size  = LoadIndex(is); 
        m_output_node_size = LoadIndex(is);
        m_input_shape      = LoadIndices(is);
        m_output_shape     = LoadIndices(is);
        m_input_index.Load(is);
        m_W->Load(is);
    }


#ifdef BB_WITH_CEREAL
    template <class Archive>
    void save(Archive& archive, std::uint32_t const version) const
    {
        _super::save(archive, version);
        archive(cereal::make_nvp("input_node_size",  m_input_node_size));
        archive(cereal::make_nvp("output_node_size", m_output_node_size));
        archive(cereal::make_nvp("input_shape",      m_input_shape));
        archive(cereal::make_nvp("output_shape",     m_output_shape));
        archive(cereal::make_nvp("input_index",      m_input_index));
        archive(cereal::make_nvp("W",                *m_W));
    }

    template <class Archive>
    void load(Archive& archive, std::uint32_t const version)
    {
        _super::load(archive, version);
        archive(cereal::make_nvp("input_node_size",  m_input_node_size));
        archive(cereal::make_nvp("output_node_size", m_output_node_size));
        archive(cereal::make_nvp("input_shape",      m_input_shape));
        archive(cereal::make_nvp("output_shape",     m_output_shape));
        archive(cereal::make_nvp("input_index",      m_input_index));
        archive(cereal::make_nvp("W",                *m_W));
    }

    void Save(cereal::JSONOutputArchive& archive) const
    {
        archive(cereal::make_nvp("RealLut2", *this));
    }

    void Load(cereal::JSONInputArchive& archive)
    {
        archive(cereal::make_nvp("RealLut2", *this));
    }
#endif


    Tensor       &W(void)       { return *m_W; }
    Tensor const &W(void) const { return *m_W; }
    
    Tensor       &dW(void)       { return *m_dW; }
    Tensor const &dW(void) const { return *m_dW; }

    auto lock_InputIndex(void)             { return m_input_index.Lock(); }
    auto lock_InputIndex_const(void) const { return m_input_index.LockConst(); }

    auto lock_W(void)              { return m_W->Lock<T>(); }
    auto lock_W_const(void) const  { return m_W->LockConst<T>(); }
    auto lock_dW(void)             { return m_dW->Lock<T>(); }
    auto lock_dW_const(void) const { return m_dW->LockConst<T>(); }


    index_t GetNodeInputSize(index_t node) const
    {
        return 2;
    }

    void SetNodeInput(index_t node, index_t input_index, index_t input_node)
    {
        auto ptr = lock_InputIndex();
        ptr(node, input_index) = (std::int32_t)input_node;
    }

    index_t GetNodeInput(index_t node, index_t input_index) const
    {
        auto ptr = lock_InputIndex_const();
        return (index_t)ptr(node, input_index);
    }


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
        m_input_node_size = GetShapeSize(shape);
        
        // 接続初期化
        m_input_index.Resize(m_output_node_size, 2);
        this->InitializeNodeInput(m_mt());

        // パラメータ初期化
        m_W->Resize(DataType<T>::type, m_output_node_size, 16);  m_W->InitUniformDistribution(0.45, 0.55, m_mt());
        m_dW->Resize(DataType<T>::type, m_output_node_size, 16); m_dW->FillZero();

        return m_output_shape;
    }

    /**
     * @brief  出力のshape設定
     * @detail 出力のshape設定
     *         出力ノード数が変わらない限りshpeは自由
     * @param shape 新しいshape
     * @return なし
     */
    void SetOutputShape(indices_t const &shape)
    {
        BB_ASSERT(GetShapeSize(shape) == m_output_node_size);
        m_output_shape = shape;
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
    
    
    
    Variables GetParameters(void)
    {
        Variables parameters;
        if ( !m_parameter_lock ) {
            parameters.PushBack(m_W);
        }
        return parameters;
    }

    Variables GetGradients(void)
    {
        Variables gradients;
        if ( !m_parameter_lock ) {
            gradients.PushBack(m_dW);
        }
        return gradients;
    }
    

    // ノード単位でのForward計算
    std::vector<double> ForwardNode(index_t node, std::vector<double> input_value) const
    {
        /*
        auto W = lock_W_const();

        */

        return input_value;
    }

    FrameBuffer Forward(FrameBuffer x, bool train = true)
    {
        BB_ASSERT(x.GetType() == DataType<T>::type);

        // backwardの為に保存
        m_x = x;

        // SetInputShpaeされていなければ初回に設定
        if (m_x.GetNodeSize() != m_input_node_size) {
            SetInputShape(m_x.GetShape());
        }

        // 出力を設定
        m_y.Resize(DataType<T>::type, m_x.GetFrameSize(), m_output_shape);

        // バイナリモードならパラメータクリップ
        if ( m_binary_mode ) {
            m_W->Clamp(0.0, 1.0);
        }

        {
            auto frame_size = m_x.GetFrameSize();
            auto x_ptr = x.LockConst<T>();
            auto y_ptr = m_y.Lock<T>();
            auto input_index_ptr = m_input_index.LockConst();
            auto W_ptr = lock_W_const();

    //#pragma omp parallel for
            for ( index_t node = 0; node < m_output_node_size; ++node ) {
                index_t in_idx[2];
                for ( int i = 0; i < 2; ++i) {
                    in_idx[i] = input_index_ptr(node, i);
                }
                T W[4];
                for ( int i = 0; i < 4; ++i) {
                    W[i] = W_ptr(node, i);
                    if ( m_binary_mode ) {
                        W[i] = W[i] > (T)0.5 ? (T)1.0 : (T)0.0;
                    }
                }

                for (index_t frame = 0; frame < frame_size; ++frame ) {
                    T   xp[2], xn[2];
                    for ( int i = 0; i < 2; ++i) {
                        xp[i] = x_ptr.Get(frame, in_idx[i]);
                        xn[i] = (T)1.0 - xp[i];
                    }

                    T xi[4];
                    xi[0]  = xn[1] * xn[0];
                    xi[1]  = xn[1] * xp[0];
                    xi[2]  = xp[1] * xn[0];
                    xi[3]  = xp[1] * xp[0];
 
                    T sig = 0;
                    for ( int i = 0; i < 4; ++i) {
                        sig += W[i] * xi[i];
                    }

                    y_ptr.Set(frame, node, sig);
                }
            }

            return m_y;
        }
    }


    FrameBuffer Backward(FrameBuffer dy)
    {
        BB_ASSERT(dy.GetType() == DataType<T>::type);

        m_dx.Resize(DataType<T>::type, dy.GetFrameSize(), m_input_node_size);
        m_dx.FillZero();
        m_dW->FillZero();

        auto frame_size = m_x.GetFrameSize();
        auto x_ptr = m_x.LockConst<T>();
        auto dy_ptr = dy.LockConst<T>();
        auto dx_ptr = m_dx.Lock<T>();
        auto input_index_ptr = m_input_index.LockConst();
        auto W_ptr  = lock_W_const();
        auto dW_ptr = lock_dW();
        
        for ( index_t node = 0; node < m_output_node_size; ++node ) {
            index_t in_idx[2];
            for ( int i = 0; i < 2; ++i) {
                in_idx[i] = input_index_ptr(node, i);
            }
            T W[4];
            for ( int i = 0; i < 4; ++i) {
                W[i] = W_ptr(node, i);
                if ( m_binary_mode ) {
                    W[i] = W[i] > (T)0.5 ? (T)1.0 : (T)0.0;
                }
            }

            T dW[4] = {0};
            for (index_t frame = 0; frame < frame_size; ++frame ) {
                T   xp[2], xn[2];
                for ( int i = 0; i < 2; ++i) {
                    xp[i] = x_ptr.Get(frame, in_idx[i]);
                    xn[i] = (T)1.0 - xp[i];
                }

                T xi[4];
                xi[0]  = xn[1] * xn[0];
                xi[1]  = xn[1] * xp[0];
                xi[2]  = xp[1] * xn[0];
                xi[3]  = xp[1] * xp[0];
                

                T dy = dy_ptr.Get(frame, node);

                T dxi[4];
                for ( int i = 0; i < 4; ++i) {
                    dW[i]  += xi[i] * dy;
                    dxi[i]  = W[i]  * dy;
                }

                T dxn[2] = {0};
                T dxp[2] = {0};
                dxn[0] += xn[1] * dxi[0]; dxn[1] += xn[0] * dxi[0];
                dxp[0] += xn[1] * dxi[1]; dxn[1] += xp[0] * dxi[1];
                dxn[0] += xp[1] * dxi[2]; dxp[1] += xn[0] * dxi[2];
                dxp[0] += xp[1] * dxi[3]; dxp[1] += xp[0] * dxi[3];

                T dx[2];
                dx[0] = (dxp[0] - dxn[0]);
                dx[1] = (dxp[1] - dxn[1]);

                for ( int i = 0; i < 2; ++i) {
                    dx_ptr.Add(frame, in_idx[i], dx[i]);
                }
            }

            for ( int i = 0; i < 4; ++i) {
                dW_ptr(node, i) = dW[i];
            }
        }

        return m_dx;
    }
};


}
