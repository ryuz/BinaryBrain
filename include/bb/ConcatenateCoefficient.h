// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                Copyright (C) 2018-2019 by Ryuji Fuchikami
//                                https://github.com/ryuz
//                                ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once


#include "bb/Manager.h"
#include "bb/Model.h"


namespace bb {



// 定数を連結する
template <typename FT = float, typename BT = float>
class ConcatenateCoefficient : public Model
{
protected:
    bool                      m_host_only   = false;
    bool                      m_binary_mode = true;

    indices_t                 m_input_shape;
    indices_t                 m_output_shape;
    indices_t                 m_coeff_shape;

    index_t                   m_concatenate_size;

    std::shared_ptr<Tensor>   m_param;
    std::shared_ptr<Tensor>   m_grad;

    FrameBuffer               m_y_buf;
    FrameBuffer               m_dx_buf;

    std::mt19937_64           m_mt;

public:
    struct create_t {
        index_t         concatenate_size = 0;
        std::uint64_t   seed = 1;
    };

protected:
    ConcatenateCoefficient(create_t const &create)
    {
        m_concatenate_size = create.concatenate_size;
        m_mt.seed(create.seed);

        m_param = std::shared_ptr<Tensor>(new Tensor);
        m_grad  = std::shared_ptr<Tensor>(new Tensor);
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


public:
    ~ConcatenateCoefficient() {}
    
    static std::shared_ptr<ConcatenateCoefficient> Create(create_t const &create)
    {
        return std::shared_ptr<ConcatenateCoefficient>(new ConcatenateCoefficient(create));
    }

    static std::shared_ptr<ConcatenateCoefficient> Create(index_t concatenate_size, std::uint64_t seed = 1)
    {
        create_t create;
        create.concatenate_size = concatenate_size;
        create.seed = seed;
        return std::shared_ptr<ConcatenateCoefficient>(new ConcatenateCoefficient(create));
    }
    
    std::string GetModelName(void) const { return "ConcatenateCoefficient"; }

    /**
     * @brief  入力のshape設定
     * @detail 入力のshape設定
     * @param shape 新しいshape
     * @return なし
     */
    indices_t SetInputShape(indices_t shape)
    {
        // 設定済みなら何もしない
        if ( shape == this->GetInputShape() ) {
            return this->GetOutputShape();
        }

        BB_ASSERT(shape.size() > 0);

        // 形状設定
        m_input_shape = shape;

        // 最後の次元にサイズを積み増しして出力形状設定
        m_output_shape = m_input_shape;
        m_output_shape[m_output_shape.size() - 1] += m_concatenate_size;

        m_coeff_shape = m_input_shape;
        m_coeff_shape[m_output_shape.size() - 1] = m_concatenate_size;


        // パラメータ初期化
        m_param->Resize(DataType<BT>::type, m_coeff_shape); *m_param = 0.5;
        m_grad->Resize(DataType<BT>::type, m_coeff_shape);  *m_grad  = 0;

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
    
    
    
    Variables GetParameters(void)
    {
        Variables parameters;
        parameters.PushBack(m_param);
        return parameters;
    }

    Variables GetGradients(void)
    {
        Variables gradients;
        gradients.PushBack(m_grad);
        return gradients;
    }
    
    void SetParameter(index_t index, BT coeff)
    {
        auto param_ptr = m_param->Lock<BT>();
        param_ptr[index] = coeff;
    }

    void GetParameter(index_t index) const
    {
        auto param_ptr = m_param->LockConst<BT>();
        return param_ptr[index];
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
        BB_ASSERT(x_buf.GetType() == DataType<FT>::type);

        // パラメータクリップ
        if ( m_binary_mode ) {
            m_param->Clamp(0.0, 1.0);
        }

        // 戻り値のサイズ設定
        m_y_buf.Resize(x_buf.GetType(), x_buf.GetFrameSize(), m_output_shape);

#if 0 // #ifdef BB_WITH_CUDA
        if ( DataType<FT>::type == BB_TYPE_FP32 && !m_host_only && m_x_buf.IsDeviceAvailable() && m_y_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
            // CUDA版
            auto ptr_x = x_buf.LockDeviceMemoryConst();
            auto ptr_y = m_y_buf.LockDeviceMemory(true);
            bbcu_fp32_ConcatenateConstant_Forward(
                        (float const *)ptr_x.GetAddr(),
                        (float       *)ptr_y.GetAddr(),
                        (int          )m_x_buf.GetNodeSize(),
                        (int          )m_x_buf.GetFrameSize(),
                        (int          )(m_x_buf.GetFrameStride() / sizeof(float))
                    );
            return m_y_buf;
        }
#endif

        {
            // 汎用版
            index_t input_node_size  = x_buf.GetNodeSize();
            index_t output_node_size = m_y_buf.GetNodeSize();
            index_t frame_size       = m_y_buf.GetFrameSize();

            auto x_ptr     = x_buf.LockConst<FT>();
            auto y_ptr     = m_y_buf.Lock<FT>();
            auto param_ptr = m_param->Lock<BT>();

            #pragma omp parallel for
            for (index_t node = 0; node < input_node_size; ++node) {
                for (index_t frame = 0; frame < frame_size; ++frame) {
                    auto x = x_ptr.Get(frame, node);
                    y_ptr.Set(frame, node, x);
                }
            }

            index_t coeff_size = output_node_size - input_node_size;
            if ( DataType<FT>::type == BB_TYPE_BIT ) {
                std::uniform_real_distribution<BT>  dist((BT)0, (BT)1);
                for (index_t i = 0; i < coeff_size; ++i) {
                    auto coeff = param_ptr[i];
                    for (index_t frame = 0; frame < frame_size; ++frame) {
                        y_ptr.Set(frame, input_node_size + i, dist(m_mt) < coeff);
                    }
                }
            }
            else {
                #pragma omp parallel for
                for (index_t i = 0; i < coeff_size; ++i) {
                    auto coeff = param_ptr[i];
                    for (index_t frame = 0; frame < frame_size; ++frame) {
                        y_ptr.Set(frame, input_node_size + i, (FT)coeff);
                    }
                }
            }

            return m_y_buf;
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
        BB_ASSERT(dy_buf.GetType() == DataType<BT>::type);

        // 戻り値のサイズ設定
        m_dx_buf.Resize(DataType<BT>::type, dy_buf.GetFrameSize(), m_input_shape);

#if 0 // #ifdef BB_WITH_CUDA
        if ( DataType<T>::type == BB_TYPE_FP32 && !m_host_only
            && m_x_buf.IsDeviceAvailable() && m_dx_buf.IsDeviceAvailable() && dy_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
            // GPU版
            auto ptr_x  = m_x_buf.LockDeviceMemoryConst();
            auto ptr_dy = dy_buf.LockDeviceMemoryConst();
            auto ptr_dx = m_dx_buf.LockDeviceMemory(true);
            bbcu_fp32_ReLU_Backward(
                        (float const *)ptr_x.GetAddr(),
                        (float const *)ptr_dy.GetAddr(),
                        (float       *)ptr_dx.GetAddr(),
                        (int          )dy_buf.GetNodeSize(),
                        (int          )dy_buf.GetFrameSize(),
                        (int          )(dy_buf.GetFrameStride() / sizeof(float))
                    );
            return m_dx_buf;
        }
#endif

        {
            //汎用版
            index_t input_node_size  = m_dx_buf.GetNodeSize();
            index_t output_node_size = dy_buf.GetNodeSize();
            index_t frame_size       = dy_buf.GetFrameSize();

            auto dy_ptr   = dy_buf.LockConst<BT>();
            auto dx_ptr   = m_dx_buf.Lock<BT>();
            auto grad_ptr = m_grad->Lock<BT>();

            #pragma omp parallel for
            for (index_t node = 0; node < input_node_size; ++node) {
                for (index_t frame = 0; frame < frame_size; ++frame) {
                    auto dy = dy_ptr.Get(frame, node);
                    dx_ptr.Set(frame, node, dy);
                }
            }

            index_t coeff_size = output_node_size - input_node_size;
            #pragma omp parallel for
            for (index_t i = 0; i < coeff_size; ++i) {
                BT dy = 0;
                for (index_t frame = 0; frame < frame_size; ++frame) {
                    dy += dy_ptr.Get(frame, input_node_size + i);
                }
                grad_ptr[i] += dy / (BT)frame_size;
            }

            return m_dx_buf;
        }
    }
};


}


// end of file