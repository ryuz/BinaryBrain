// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                Copyright (C) 2018-2019 by Ryuji Fuchikami
//                                https://github.com/ryuz
//                                ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once


#include "bb/Manager.h"
#include "bb/Activation.h"


namespace bb {


// Sigmoid(活性化層)
template <typename T = float>
class Softmax: public Activation
{
    using _super = Activation;

public:
    static inline std::string ModelName(void) { return "Softmax"; }
    static inline std::string ObjectName(void){ return ModelName() + "_" + DataType<T>::Name(); }

    std::string GetModelName(void)  const override { return ModelName(); }
    std::string GetObjectName(void) const override { return ObjectName(); }

protected:
    indices_t   m_shape;
    bool        m_host_only = false;

    FrameBuffer m_y_buf;

protected:
    Softmax() {
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


public:
    static std::shared_ptr<Softmax> Create(void)
    {
        return std::shared_ptr<Softmax>(new Softmax);
    }

    ~Softmax() {}
    

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
        // 設定済みなら何もしない
        if ( shape == this->GetInputShape() ) {
            return this->GetOutputShape();
        }

        m_shape = shape;
        return m_shape;
    }

    /**
     * @brief  入力形状取得
     * @detail 入力形状を取得する
     * @return 入力形状を返す
     */
    indices_t GetInputShape(void) const override
    {
        return m_shape;
    }

    /**
     * @brief  出力形状取得
     * @detail 出力形状を取得する
     * @return 出力形状を返す
     */
    indices_t GetOutputShape(void) const override
    {
        return m_shape;
    }

    // 1ノードのみForward計算
    std::vector<double> ForwardNode(index_t node, std::vector<double> x_vec) const
    {
        std::vector<double> y_vec(x_vec.size());
        double max_x = 0;
        for ( auto x : x_vec ) {
            max_x = std::max(max_x, x);
        }

        double sum = 0;
        for ( size_t i=0;  i < x_vec.size(); ++i ) {
            auto x = x_vec[i] - max_x;
            y_vec[i] = std::exp(x);
            sum += y_vec[i];
        }
        sum = std::max(sum, 1.0e-7);
        for ( auto& y : y_vec ) {
            y /= sum;
        }

        return y_vec;
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
        // binaryモード
        BB_ASSERT(x_buf.GetType() == DataType<T>::type);

        // 戻り値のサイズ設定
        FrameBuffer y_buf(x_buf.GetFrameSize(), x_buf.GetShape(), x_buf.GetType());

        // ローカルに保存
        if ( train ) {
            m_y_buf = y_buf;
        }
        
        {
            // 汎用版
            auto frame_size  = x_buf.GetFrameSize();
            auto node_size   = x_buf.GetNodeSize();
            
            auto shape    = x_buf.GetShape();
            auto ch_size  = shape[0];
            auto pix_size = node_size / ch_size;

            auto x_ptr = x_buf.LockConst<T>();
            auto y_ptr = y_buf.Lock<T>(true);

//          #pragma omp parallel for
            for (index_t frame = 0; frame < frame_size; ++frame) {
                for (index_t pix = 0; pix < pix_size; ++pix) {
                    // max
                    T   c = 0;
                    for (index_t ch = 0; ch < ch_size; ++ch) {
                        auto node = ch * pix_size + pix;
                        c = std::max(c, x_ptr.Get(frame, node));
                    }

                    // sum(exp(x - c))
                    T sum = 0;
                    for (index_t ch = 0; ch < ch_size; ++ch) {
                        auto node = ch * pix_size + pix;
                        sum += std::exp(x_ptr.Get(frame, node) - c);
                    }
                    sum = std::max(sum, (T)1.0e-7);

                    for (index_t ch = 0; ch < ch_size; ++ch) {
                        auto node = ch * pix_size + pix;
                        auto y = std::exp(x_ptr.Get(frame, node) - c) / sum;
                        y_ptr.Set(frame, node, y);
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
    inline FrameBuffer Backward(FrameBuffer dy_buf) override
    {
        BB_ASSERT(dy_buf.GetType() == DataType<T>::type);

        // 戻り値のサイズ設定
        FrameBuffer dx_buf(dy_buf.GetFrameSize(), dy_buf.GetShape(), dy_buf.GetType());

        FrameBuffer y_buf = m_y_buf;
        m_y_buf = FrameBuffer();

        {
            auto frame_size  = dy_buf.GetFrameSize();
            auto node_size   = dy_buf.GetNodeSize();
            
            auto shape    = dy_buf.GetShape();
            auto ch_size  = shape[0];
            auto pix_size = node_size / ch_size;

            auto y_ptr  = y_buf.LockConst<T>();
            auto dy_ptr = dy_buf.LockConst<T>();
            auto dx_ptr = dx_buf.Lock<T>(true);

//          #pragma omp parallel for
            for (index_t frame = 0; frame < frame_size; ++frame) {
                for (index_t pix = 0; pix < pix_size; ++pix) {
                    // sum(dy*y)
                    T sum = 0;
                    for (index_t ch = 0; ch < ch_size; ++ch) {
                        auto node = ch * pix_size + pix;
                        auto dy = dy_ptr.Get(frame, node);
                        auto y  = y_ptr.Get(frame, node);
                        sum += dy * y;
                    }

                    for (index_t ch = 0; ch < ch_size; ++ch) {
                        auto node = ch * pix_size + pix;
                        auto dy = dy_ptr.Get(frame, node);
                        auto y  = y_ptr.Get(frame, node);
                        dx_ptr.Set(frame, node, (dy - sum)*y);
                    }
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
    }

    void LoadObjectData(std::istream &is) override
    {
        // バージョン
        std::int64_t ver;
        bb::LoadValue(is, ver);

        BB_ASSERT(ver == 1);

        // 親クラス
        _super::LoadObjectData(is);
    }

};


}


