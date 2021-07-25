// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once

#include <vector>
#include <stack>

#include "bb/Manager.h"
#include "bb/Model.h"


namespace bb {


// Concatenate
class Concatenate : public Model
{
    using _super = Model;

public:
    static inline std::string ClassName(void) { return "Concatenate"; }
    static inline std::string ObjectName(void){ return ClassName(); }

    std::string GetModelName(void)  const { return ClassName(); }
    std::string GetObjectName(void) const { return ObjectName(); }

protected:
    bool                                    m_host_only = false;
    std::vector<indices_t>                  m_input_shapes;
    indices_t                               m_output_shape;

    std::stack< std::vector<indices_t> >    m_shapes;
    
public:
    
protected:
    Concatenate() {}

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
    ~Concatenate() {}

    static std::shared_ptr<Concatenate> Create(void)
    {
        return std::shared_ptr<Concatenate>(new Concatenate());
    }
    
    indices_t SetInputShape(indices_t shape) override
    {
        auto shapes = SetInputShapeMulti({shape});
        BB_ASSERT(shapes.size()==1);
        return shapes[0];
    }

    std::vector<indices_t> SetInputShapeMulti(std::vector<indices_t> shapes) override
    {
        BB_ASSERT(shapes.size() > 0);
        for (int i = 1; i < (int)shapes.size(); ++i) {
            BB_ASSERT(shapes[i].size() > 0);
            BB_ASSERT(shapes[i].size() == shapes[0].size());
            for (int c = 1; c < (int)shapes[0].size(); ++c) {
                BB_ASSERT(shapes[i][c] == shapes[0][c]);
            }
        }

        m_input_shapes = shapes;
        m_output_shape = m_input_shapes[0];
        for (int i = 1; i < (int)m_input_shapes.size(); ++i) {
            m_output_shape[0] += m_input_shapes[i][0];
        }

        return {m_output_shape};
    }

    indices_t GetInputShape(void) const override
    {
        return m_input_shapes[0];
    }

    indices_t GetOutputShape(void) const override
    {
        return m_output_shape;
    }


    inline void Clear(void) override
    {
        while ( !m_shapes.empty() ) { m_shapes.pop(); }
    }

    virtual FrameBuffer Forward(FrameBuffer x_buf, bool train = true) override
    {
        return ForwardMulti({x_buf}, train)[0];
    }


    /**
     * @brief  forward演算
     * @detail forward演算を行う
     * @param  x     入力データ
     * @param  train 学習時にtrueを指定
     * @return forward演算結果
     */
    virtual std::vector<FrameBuffer> ForwardMulti(std::vector<FrameBuffer> x_bufs, bool train = true) override
    {
        BB_ASSERT(x_bufs.size() > 0);
        auto data_type    = x_bufs[0].GetType();
        auto frame_size   = x_bufs[0].GetFrameSize();
        auto frame_stride = x_bufs[0].GetFrameStride();

        for ( int i = 1; i < (int)x_bufs.size(); ++i ) {
            BB_ASSERT(x_bufs[i].GetType()        == data_type);
            BB_ASSERT(x_bufs[i].GetFrameSize()   == frame_size);
            BB_ASSERT(x_bufs[i].GetFrameStride() == frame_stride);
        }

        std::vector<indices_t>  x_shapes;
        for ( auto& x_buf : x_bufs ) {
            x_shapes.push_back(x_buf.GetShape());
        }
        
        auto y_shape = SetInputShapeMulti(x_shapes)[0];

        // backwardの為に保存
        if ( train ) {
            m_shapes.push(x_shapes);
        }

        // 戻り値のサイズ設定
        FrameBuffer y_buf(frame_size, y_shape, data_type);


#ifdef BB_WITH_CUDA
        bool x_device_available = true;
        for ( auto& x_buf : x_bufs ) {
            if (!x_buf.IsDeviceAvailable()) {
                x_device_available = false;
            }
        }

        if ( !m_host_only && x_device_available && y_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
            // CUDA版
            auto ptr_y = y_buf.LockDeviceMemory(true);
            char *adr_y = (char *)ptr_y.GetAddr();
            for ( auto& x_buf : x_bufs ) {
                auto ptr_x = x_buf.LockDeviceMemoryConst();
                auto adr_x = ptr_x.GetAddr();

                auto copy_size = frame_stride * x_buf.GetNodeSize();
                cudaMemcpy(adr_y, adr_x, (size_t)copy_size, cudaMemcpyDeviceToDevice);
                adr_y += copy_size;
            }

            return {y_buf};
        }
#endif

        {
            // 汎用版
            auto ptr_y = y_buf.Lock<int>(true);
            char *adr_y = (char *)ptr_y.GetAddr();
            for ( auto& x_buf : x_bufs ) {
                auto ptr_x = x_buf.LockConst<int>();
                auto adr_x = ptr_x.GetAddr();

                auto copy_size = frame_stride * x_buf.GetNodeSize();
                memcpy(adr_y, adr_x, (size_t)copy_size);
                adr_y += copy_size;
            }

            return {y_buf};
        }
    }


    virtual FrameBuffer Backward(FrameBuffer dy_buf) override
    {
        return BackwardMulti({dy_buf})[0];
    }


   /**
     * @brief  backward演算
     * @detail backward演算を行う
     *         
     * @return backward演算結果
     */
    inline std::vector<FrameBuffer> BackwardMulti(std::vector<FrameBuffer> dy_bufs) override
    {
        BB_ASSERT(dy_bufs.size() == 1);
        auto dy_buf = dy_bufs[0];
        auto data_type    = dy_buf.GetType();
        auto frame_size   = dy_buf.GetFrameSize();
        auto frame_stride = dy_buf.GetFrameStride();

        BB_ASSERT(!m_shapes.empty());
        auto x_shapes = m_shapes.top();  m_shapes.pop();

        auto y_shape = SetInputShapeMulti(x_shapes)[0];
        BB_ASSERT(dy_buf.GetShape() == y_shape);

        // 戻り値のサイズ設定
        std::vector<FrameBuffer> dx_bufs;
        for (auto& x_shape : x_shapes) {
            dx_bufs.push_back(FrameBuffer(frame_size, x_shape, data_type));
        }


#ifdef BB_WITH_CUDA
        bool dx_device_available = true;
        for ( auto& dx_buf : dx_bufs ) {
            if (!dx_buf.IsDeviceAvailable()) {
                dx_device_available = false;
            }
        }

        if ( !m_host_only && dx_device_available && dy_buf.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
            // CUDA版
            auto ptr_y  = dy_buf.LockDeviceMemoryConst();
            auto adr_y = (char const *)ptr_y.GetAddr();
            for ( auto& dx_buf : dx_bufs ) {
                auto ptr_x = dx_buf.LockDeviceMemory(true);
                auto adr_x = ptr_x.GetAddr();

                auto copy_size = frame_stride * dx_buf.GetNodeSize();
                cudaMemcpy(adr_x, adr_y, (size_t)copy_size, cudaMemcpyDeviceToDevice);
                adr_y += copy_size;
            }

            return dx_bufs;
        }
#endif

        {
            // 汎用版
            auto ptr_y  = dy_buf.LockConst<int>();
            auto adr_y = (char const *)ptr_y.GetAddr();
            for ( auto& dx_buf : dx_bufs ) {
                auto ptr_x = dx_buf.Lock<int>(true);
                auto adr_x = ptr_x.GetAddr();

                auto copy_size = frame_stride * dx_buf.GetNodeSize();
                cudaMemcpy(adr_x, adr_y, (size_t)copy_size, cudaMemcpyDeviceToDevice);
                adr_y += copy_size;
            }

            return dx_bufs;
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
        bb::SaveValue(os, m_input_shapes);
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
        bb::LoadValue(is, m_host_only);
        bb::LoadValue(is, m_input_shapes);
        bb::LoadValue(is, m_output_shape);
    }
};


};

