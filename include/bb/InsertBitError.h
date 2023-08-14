// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once

#include <random>

#include "bb/Model.h"


namespace bb {


/**
 * @brief   バイナリデータにビットエラーをのせる
 */
template <typename BinType = float, typename RealType = float>
class InsertBitError : public Model
{
    using _super = Model;

public:
    static inline std::string ClassName(void) { return "InsertBitError"; }
    static inline std::string ObjectName(void){ return ClassName() + "_" + DataType<BinType>::Name() + "_" + DataType<RealType>::Name(); }

    std::string GetModelName(void)  const { return ClassName(); }
    std::string GetObjectName(void) const { return ObjectName(); }

protected:
    bool                m_host_only = false;
    indices_t           m_shape;
    double              m_error_rate;
    std::minstd_rand    m_rand_engine;

public:
    struct create_t
    {
        double  error_rate;     // エラー率


        void ObjectDump(std::ostream& os) const
        {
            bb::SaveValue(os, error_rate);
        }

        void ObjectLoad(std::istream& is)
        {
            bb::LoadValue(is, error_rate);
        }
    };

protected:
    InsertBitError(create_t const &create)
    {
        m_error_rate = create.error_rate;
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
    ~InsertBitError() {}


    static std::shared_ptr<InsertBitError> Create(create_t const &create)
    {
        return std::shared_ptr<InsertBitError>(new InsertBitError(create));
    }

    static std::shared_ptr<InsertBitError> Create(double error_rate)
    {
        create_t create;
        create.error_rate = error_rate;
        return Create(create);
    }

    /**
     * @brief  入力のshape設定
     * @detail 入力のshape設定
     * @param shape 新しいshape
     * @return なし
     */
    indices_t SetInputShape(indices_t shape) override
    {
        // 設定済みなら何もしない
        if (shape == this->GetInputShape()) {
            return this->GetOutputShape();
        }

        // 形状設定
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
    indices_t GetOutputShape(void) const
    {
        return m_shape;
    }
    

    FrameBuffer Forward(FrameBuffer x_buf, bool train = true)
    {
        if (!train) {
            return x_buf;
        }

        BB_ASSERT(x_buf.GetType() == DataType<BinType>::type);

        // SetInputShpaeされていなければ初回に設定
        if (x_buf.GetShape() != m_shape) {
            (void)SetInputShape(x_buf.GetShape());
        }

        index_t frame_size = x_buf.GetFrameSize();
        index_t node_size = x_buf.GetNodeSize();

        // 出力を設定
        FrameBuffer y_buf(frame_size, this->GetOutputShape(), DataType<BinType>::type);


        // エラーテーブルを生成
        auto error_buf = FrameBuffer(frame_size, m_shape, BB_TYPE_BIT);
        {
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            auto err_ptr = error_buf.template Lock<Bit>();
            for (index_t frame = 0; frame < frame_size; ++frame) {
                for (index_t node = 0; node < node_size; ++node) {
                    err_ptr.Set(frame, node, dist(m_rand_engine) < m_error_rate);
                }
            }
        }
        // backwardの為に保存
        this->PushFrameBuffer(error_buf);


        {
            // 汎用版
            auto err_ptr = error_buf.template LockConst<BinType>();
            auto x_ptr   = x_buf.template LockConst<BinType>();
            auto y_ptr   = y_buf.template Lock<BinType>();

            #pragma omp parallel for
            for (index_t frame = 0; frame < frame_size; ++frame) {
                for (index_t node = 0; node < node_size; ++node) {
                    auto x = x_ptr.Get(frame, node);
                    if (err_ptr.Get(frame, node)) {
                        y_ptr.Set(frame, node, (BinType)1 - x);
                    }
                    else {
                        y_ptr.Set(frame, node, x);
                    }
                }
            }
            return y_buf;
        }
    }


    FrameBuffer Backward(FrameBuffer dy_buf)
    {
        auto error_buf = this->PopFrameBuffer();

        if (error_buf.Empty()) {
            return FrameBuffer();
        }


        BB_ASSERT(dy_buf.GetType() == DataType<RealType>::type);

        index_t frame_size = dy_buf.GetFrameSize();
        index_t node_size = dy_buf.GetNodeSize();

        // 戻り値の型を設定
        FrameBuffer dx_buf(frame_size, dy_buf.GetShape(), DataType<RealType>::type);


        {
            // 汎用版
            auto err_ptr = error_buf.template LockConst<BinType>();
            auto dy_ptr = dy_buf.template LockConst<RealType>();
            auto dx_ptr = dx_buf.template Lock<RealType>();

#pragma omp parallel for
            for (index_t frame = 0; frame < frame_size; ++frame) {
                for (index_t node = 0; node < node_size; ++node) {
                    auto dy = dy_ptr.Get(frame, node);
                    if (err_ptr.Get(frame, node)) {
                        dx_ptr.Set(frame, node, -dy);
                    }
                    else {
                        dx_ptr.Set(frame, node, dy);
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

        // メンバ
        bb::SaveValue(os, m_error_rate);
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
        bb::LoadValue(is, m_error_rate);
    }
};


}


// end of file
