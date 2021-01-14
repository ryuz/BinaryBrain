// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once

#include <mutex>
#include <iostream>
#include <fstream>
#include <array>
#include <vector>
#include <memory>
#include <malloc.h>

#include "bb/DataType.h"
#include "bb/Object.h"
#include "bb/Tensor.h"


namespace bb {


// [FrameBuffer クラス]
//   ・LayerとLayerの接続に利用
//   ・Layerのパラメータ構成に必要な情報を保持
//   ・Tensorクラスで実体を保有
//   ・Tensor内のデータの各次元の意味付けを行う(Sparseで性能の出る軸で管理)
// 
//  外部APIとしては、Tensor が複数フレーム格納されるたような形式に見せる
//  内部的には１つの Tensor に統合する。即ち内部 Tensor は次数が1多い
//  メモリ配置は NCHW でも NHWC でもなく、CHWN を意図しており、N = frame の意である
//
//  また ここで node_size や node などの語を定義している。これは各フレームの
//  Tensor を1次元のフラットでアクセスする事を意図した用語である。
//  CUDAやSIMD命令での操作を強く意図しており、これらを使ってプログラミングするときは
//  メモリ配置を強く意識する必要がある。 shape は上位からメモリにアクセスをする際に
//  利便性を向上させる為のものである。
//  同じノードへのアクセス方法として、
//    ・フレーム番号＋ノード番号
//    ・フレーム番号＋shapeに従った多次元の添え字
//  の２種類があるので注意すること



// -------------------------------------
//  アクセス用ポインタクラス定義
// -------------------------------------

class FrameBuffer;

// const アクセス用
template <typename Tp, class FrameBufferTp, class PtrTp>
class FrameBufferConstPtr_
{
friend FrameBufferTp;

protected:
    FrameBufferTp*   m_buf;
    PtrTp            m_ptr;

protected:
    FrameBufferConstPtr_(FrameBufferTp* buf)
    {
        m_buf = buf;
    }

public:
    FrameBufferConstPtr_(FrameBufferConstPtr_ const &buf)
    {
        m_buf = buf.m_buf;
        m_ptr = buf.m_ptr;
    }

protected:
    
    inline void Lock(void)
    {
        m_ptr = m_buf->LockMemoryConst();
    }

    inline void const *GetNodeBaseAddr(index_t node) const
    {
        auto addr = (std::uint8_t const *)m_ptr.GetAddr();
        return addr + (m_buf->m_frame_stride * node);
    }

    inline index_t GetNodeIndex(indices_t const & indices) const
    {
        return ConvertIndicesToIndex(indices, m_buf->GetShape());
    }

    inline index_t GetNodeIndex(index_t i0) const
    {
        return ConvertIndicesToIndex(i0, m_buf->GetShape());
    }

    inline index_t GetNodeIndex(index_t i0, index_t i1) const
    {
        return ConvertIndicesToIndex(i0, i1, m_buf->GetShape());
    }

    inline index_t GetNodeIndex(index_t i0, index_t i1, index_t i2) const
    {
        return ConvertIndicesToIndex(i0, i1, i2, m_buf->GetShape());
    }

    inline Tp ReadValue(void const *base, index_t frame) const
    {
        return DataType_Read<Tp>(base, frame);
    }


public:
    inline FrameBuffer const &GetFrameBuffer(void) const
    {
        return *m_buf;
    }

    inline Tp const *GetAddr(void) const
    {
        return (Tp *)m_ptr.GetAddr();
    }

    inline Tp const *GetAddr(index_t node) const
    {
        return (Tp *)GetNodeBaseAddr(node);
    }

    inline Tp Get(index_t frame, index_t node) const
    {
        return ReadValue(GetNodeBaseAddr(node),  frame);
    }

    inline Tp Get(index_t frame, indices_t indices) const
    {
        return Get(frame, GetNodeIndex(indices));
    }

    inline Tp Get(index_t frame, index_t i0, index_t i1) const
    {
        return Get(frame, GetNodeIndex(i0, i1));
    }

    inline Tp Get(index_t frame, index_t i0, index_t i1, index_t i2) const
    {
        return Get(frame, GetNodeIndex(i0, i1, i2));
    }
};


// 非const アクセス用
template <typename Tp, class FrameBufferTp, class PtrTp>
class FrameBufferPtr_ : public FrameBufferConstPtr_<Tp, FrameBufferTp, PtrTp>
{
    friend FrameBufferTp;
    using FrameBufferConstPtr_<Tp, FrameBufferTp, PtrTp>::m_buf;
    using FrameBufferConstPtr_<Tp, FrameBufferTp, PtrTp>::m_ptr;

protected:
    FrameBufferPtr_(FrameBufferTp* buf) : FrameBufferConstPtr_<Tp, FrameBufferTp, PtrTp>(buf)
    {
    }

    inline void Lock(bool new_buf)
    {
        m_ptr = m_buf->LockMemory(new_buf);
    }

protected:
    inline void *GetNodeBaseAddr(index_t node)
    {
        BB_DEBUG_ASSERT(node >= 0 && node < m_buf->GetNodeSize());
        auto addr = (std::uint8_t *)m_ptr.GetAddr();
        return addr + (m_buf->m_frame_stride * node);
    }

    inline index_t GetNodeIndex(indices_t const & indices)
    {
        return ConvertIndicesToIndex(indices, m_buf->m_node_shape);
    }

    inline index_t GetNodeIndex(index_t i0)
    {
        return ConvertIndicesToIndex(i0, m_buf->m_node_shape);
    }

    inline index_t GetNodeIndex(index_t i0, index_t i1)
    {
        return ConvertIndicesToIndex(i0, i1, m_buf->m_node_shape);
    }

    inline index_t GetNodeIndex(index_t i0, index_t i1, index_t i2)
    {
        return ConvertIndicesToIndex(i0, i1, i2, m_buf->m_node_shape);
    }

    inline void WriteValue(void *base, index_t frame, Tp value)
    {
        return DataType_Write<Tp>(base, frame, value);
    }

    inline void AddValue(void *base, index_t frame, Tp value)
    {
        return DataType_Add<Tp>(base, frame, value);
    }


public:
    inline Tp *GetAddr(void)
    {
        return (Tp *)m_ptr.GetAddr();
    }

    inline Tp *GetAddr(index_t node)
    {
        return (Tp *)GetNodeBaseAddr(node);
    }

    inline void Set(index_t frame, index_t node, Tp value)
    {
        return WriteValue(GetNodeBaseAddr(node),  frame, value);
    }

    inline void Set(index_t frame, indices_t indices, Tp value)
    {
        return Set(frame, GetNodeIndex(indices), value);
    }

    inline void Set(index_t frame, index_t i0, index_t i1, Tp value)
    {
        return Set(frame, GetNodeIndex(i0, i1), value);
    }

    inline void Set(index_t frame, index_t i0, index_t i1, index_t i2, Tp value)
    {
        return Set(frame, GetNodeIndex(i0, i1, i2), value);
    }


    inline void Add(index_t frame, index_t node, Tp value)
    {
        return AddValue(GetNodeBaseAddr(node),  frame, value);
    }

    inline void Add(index_t frame, indices_t indices, Tp value)
    {
        return Add(frame, GetNodeIndex(indices), value);
    }

    inline void Add(index_t frame, index_t i0, index_t i1, Tp value)
    {
        return Add(frame, GetNodeIndex(i0, i1), value);
    }

    inline void Add(index_t frame, index_t i0, index_t i1, index_t i2, Tp value)
    {
        return Add(frame, GetNodeIndex(i0, i1, i2), value);
    }
};



// NeuralNet用のバッファ
class FrameBuffer : public Object
{
    using _super = Object;

public:
    static inline std::string ObjectName(void){ return "Tensor"; }
    std::string GetObjectName(void) const override { return ObjectName(); }

protected:
    Tensor                  m_tensor;

    int                     m_data_type = 0;
    index_t                 m_frame_size = 0;
    index_t                 m_frame_stride = 0;
    index_t                 m_node_size = 0;
    indices_t               m_node_shape;

public:
    /**
      * @brief  デフォルトコンストラクタ
      * @detail デフォルトコンストラクタ
      */
//  explicit FrameBuffer(bool host_only=false) : m_tensor(host_only) {}

    /**
     * @brief  コンストラクタ
     * @detail コンストラクタ
     *        tensor は node_size サイズの1次元で初期化
     * @param frame_size フレーム数
     * @param node_size  1フレームのノード数
     * @param data_type  1ノードのデータ型
     */
//    explicit FrameBuffer(index_t frame_size, index_t node_size, int data_type, bool host_only=false) : m_tensor(host_only)
//    {
//        Resize(frame_size, indices_t({node_size}), data_type);
//    }

    /**
     * @brief  コンストラクタ
     * @detail コンストラクタ
     * @param frame_size フレーム数
     * @param shape      1フレームのノードを構成するshape
     * @param data_type  1ノードのデータ型
     */
    explicit FrameBuffer(index_t frame_size=0, indices_t shape=indices_t(), int data_type=BB_TYPE_FP32, bool host_only=false) : m_tensor(host_only)
    {
        if ( frame_size > 0 ) {
            Resize(frame_size, shape, data_type);
        }
    }

    /**
      * @brief  コピーコンストラクタ
      * @detail コピーコンストラクタ
      */
    FrameBuffer(FrameBuffer const &buf)
    {
        *this = buf;
    }

    /**
     * @brief  代入演算子
     * @detail 代入演算子
     *         代入演算子でのコピーは、メモリは同じ箇所を指す
     */
    FrameBuffer& operator=(FrameBuffer const &buf)
    {
        m_tensor        = buf.m_tensor;
        m_data_type     = buf.m_data_type;
        m_frame_size    = buf.m_frame_size;
        m_frame_stride  = buf.m_frame_stride;
        m_node_size     = buf.m_node_size;
        m_node_shape    = buf.m_node_shape;

        return *this;
    }

#ifdef BB_PYBIND11
    template<typename Tp>
    static FrameBuffer FromNumpy(pybind11::array_t<Tp> ndarray, int data_type, index_t frame_size, index_t frame_stride, indices_t node_shape, bool host_only=false)
    {
        FrameBuffer buf;
        buf.m_tensor       = Tensor_<Tp>::FromNumpy(ndarray, host_only);
        buf.m_data_type    = data_type;
        buf.m_frame_size   = frame_size;
        buf.m_frame_stride = frame_stride;
        buf.m_node_shape   = node_shape;
        buf.m_node_size    = CalcShapeSize(node_shape);
        return buf;
    }

    template<typename Tp>
    pybind11::array_t<Tp> Numpy(void)
    {
        return m_tensor.Numpy<Tp>();
    }
#endif

    /**
     * @brief  クローン
     * @detail クローン
     * @return メモリ内容をコピーしたクローンを返す
     */
    FrameBuffer Clone(void) const
    {
        FrameBuffer clone_buf;

        clone_buf.m_tensor       = m_tensor.Clone();
        clone_buf.m_data_type    = m_data_type;
        clone_buf.m_frame_size   = m_frame_size;
        clone_buf.m_frame_stride = m_frame_stride;
        clone_buf.m_node_size    = m_node_size;
        clone_buf.m_node_shape   = m_node_shape;

        return clone_buf;
    }

    bool IsHostOnly(void) const
    {
        return m_tensor.IsHostOnly();
    }
    

   /**
     * @brief  デバイスが利用可能か問い合わせる
     * @detail デバイスが利用可能か問い合わせる
     * @return デバイスが利用可能ならtrue
     */
    bool IsDeviceAvailable(void) const
    {
        return m_tensor.IsDeviceAvailable();
    }

    /**
     * @brief  サイズ設定
     * @detail サイズ設定
     * @param frame_size フレーム数
     * @param shape      1フレームのノードを構成するshape
     * @param data_type  1ノードのデータ型
     */
    static index_t CalcFrameStride(int data_type, index_t frame_size)
    {
        return ((frame_size * DataType_GetBitSize(data_type) + 255) / 256) * (256 / 8);        // frame軸は256bit境界にあわせる(SIMD命令用)
    }

    /**
     * @brief  サイズ設定
     * @detail サイズ設定
     * @param frame_size フレーム数
     * @param shape      1フレームのノードを構成するshape
     * @param data_type  1ノードのデータ型
     */
    void Resize(index_t frame_size, indices_t shape, int data_type)
    {
        m_data_type    = data_type;
        m_frame_size   = frame_size;
        m_frame_stride = CalcFrameStride(data_type, frame_size);    // frame軸は256bit境界にあわせる(SIMD命令用)
        m_node_shape   = shape;


        // Bit型は内部 UINT8 で扱う
        int tensor_type = data_type;
        if ( data_type == BB_TYPE_BIT || data_type == BB_TYPE_BINARY ) {
            tensor_type = BB_TYPE_UINT8;
        }

        // サイズ計算
        m_node_size = 1;
        std::vector<index_t>    tensor_shape;
        for ( auto size : shape ) {
            tensor_shape.push_back(size);
            m_node_size *= size;
        }
        tensor_shape.push_back(m_frame_stride / DataType_GetByteSize(tensor_type));

        // メモリ確保
        m_tensor.Resize(tensor_shape, tensor_type);
    }


    /**
     * @brief  同じ型同じサイズにする
     * @detail 同じ型同じサイズにする
     * @param buf 型とサイズをあわせる対象
     */
    void ResizeLike(FrameBuffer const &buf)
    {
        Resize(buf.GetFrameSize(), buf.GetShape(), buf.GetType());
    }
    
    template <typename T>
    void CopyTo_(FrameBuffer& dst, index_t frame_size = 0, index_t src_frame_offset = 0, index_t dst_frame_offset = 0,
                    index_t node_size = 0, index_t src_node_offset = 0, index_t dst_node_offset = 0) const
    {
         auto dst_ptr = dst.Lock<T>();
         auto src_ptr = LockConst<T>();
         for ( index_t node = 0; node < node_size; ++node) {
             for ( index_t frame = 0; frame < frame_size; ++frame) {
                 dst_ptr.Set(frame + dst_frame_offset, node + dst_node_offset, src_ptr.Get(frame + src_frame_offset, node + src_node_offset));
             }
         }
    }

    /**
     * @brief  コピー
     * @detail コピーを行う
     */
    void CopyTo(FrameBuffer& dst, index_t frame_size = 0, index_t src_frame_offset=0, index_t dst_frame_offset=0,
                index_t node_size = 0,index_t src_node_offset=0, index_t dst_node_offset=0) const
    {
        BB_ASSERT(dst.GetType() == GetType());

        if ( node_size <= 0) {
            node_size = std::min(dst.GetNodeSize() - dst_node_offset, GetFrameSize() - src_node_offset);
        }
        if ( frame_size <= 0) {
            frame_size = std::min(dst.GetFrameSize() - dst_frame_offset, GetFrameSize() - dst_frame_offset);
        }

        BB_ASSERT(frame_size + src_frame_offset <= GetFrameSize());
        BB_ASSERT(frame_size + dst_frame_offset <= dst.GetFrameSize());
        BB_ASSERT(node_size + src_node_offset <= GetNodeSize());
        BB_ASSERT(node_size + dst_node_offset <= dst.GetNodeSize());

#ifdef BB_WITH_CUDA
        if (dst.IsDeviceAvailable() && IsDeviceAvailable()) {
            if ( DataType_GetBitSize(GetType()) == 32 ) {
                auto dst_ptr = dst.LockDeviceMemory();
                auto src_ptr = LockDeviceMemoryConst();
                bbcu_int32_FrameBufferCopy
                    (
                        (int       *)dst_ptr.GetAddr(),
                        (int const *)src_ptr.GetAddr(),
                        (int        )node_size,
                        (int        )dst_node_offset,
                        (int        )src_node_offset,
                        (int        )frame_size,
                        (int        )dst_frame_offset,
                        (int        )src_frame_offset,
                        (int        )(dst.GetFrameStride() / 4),
                        (int        )(GetFrameStride() / 4)
                    );
                return;
            }

            if ( DataType_GetBitSize(GetType()) == 1 ) {
                BB_ASSERT(dst_frame_offset % 32 == 0);
                BB_ASSERT(src_frame_offset % 32 == 0);
                BB_ASSERT(frame_size % 32 == 0);

                auto dst_ptr = dst.LockDeviceMemory();
                auto src_ptr = LockDeviceMemoryConst();
                bbcu_int32_FrameBufferCopy
                    (
                        (int       *)dst_ptr.GetAddr(),
                        (int const *)src_ptr.GetAddr(),
                        (int        )node_size,
                        (int        )dst_node_offset,
                        (int        )src_node_offset,
                        (int        )(frame_size / 32),
                        (int        )(dst_frame_offset / 32),
                        (int        )(src_frame_offset / 32),
                        (int        )(dst.GetFrameStride() / 4),
                        (int        )(GetFrameStride() / 4)
                    );
                return;
            }
        }
#endif

        switch ( GetType() ) {
        case BB_TYPE_BIT:    CopyTo_<Bit     >(dst, frame_size, src_frame_offset, dst_frame_offset, node_size, src_node_offset, dst_node_offset); return;
        case BB_TYPE_FP32:   CopyTo_<float   >(dst, frame_size, src_frame_offset, dst_frame_offset, node_size, src_node_offset, dst_node_offset); return;
        case BB_TYPE_FP64:   CopyTo_<double  >(dst, frame_size, src_frame_offset, dst_frame_offset, node_size, src_node_offset, dst_node_offset); return;
        case BB_TYPE_INT8:   CopyTo_<int8_t  >(dst, frame_size, src_frame_offset, dst_frame_offset, node_size, src_node_offset, dst_node_offset); return;
        case BB_TYPE_INT16:  CopyTo_<int16_t >(dst, frame_size, src_frame_offset, dst_frame_offset, node_size, src_node_offset, dst_node_offset); return;
        case BB_TYPE_INT32:  CopyTo_<int32_t >(dst, frame_size, src_frame_offset, dst_frame_offset, node_size, src_node_offset, dst_node_offset); return;
        case BB_TYPE_INT64:  CopyTo_<int64_t >(dst, frame_size, src_frame_offset, dst_frame_offset, node_size, src_node_offset, dst_node_offset); return;
        case BB_TYPE_UINT8:  CopyTo_<uint8_t >(dst, frame_size, src_frame_offset, dst_frame_offset, node_size, src_node_offset, dst_node_offset); return;
        case BB_TYPE_UINT16: CopyTo_<uint16_t>(dst, frame_size, src_frame_offset, dst_frame_offset, node_size, src_node_offset, dst_node_offset); return;
        case BB_TYPE_UINT32: CopyTo_<uint32_t>(dst, frame_size, src_frame_offset, dst_frame_offset, node_size, src_node_offset, dst_node_offset); return;
        case BB_TYPE_UINT64: CopyTo_<uint64_t>(dst, frame_size, src_frame_offset, dst_frame_offset, node_size, src_node_offset, dst_node_offset); return;
        }
        
        // 他は必要なときに足す
        BB_ASSERT(0);
    }


    // -------------------------------------
    //  シリアライズ
    // -------------------------------------
protected:
    void DumpObjectData(std::ostream &os) const override
    {
        // バージョン
        std::int64_t ver = 1;
        bb::SaveValue(os, ver);

        // 親クラス
        _super::DumpObjectData(os);

        // メンバ
        bb::SaveValue(os, m_data_type);
        bb::SaveValue(os, m_frame_size);
        bb::SaveValue(os, m_frame_stride);
        bb::SaveValue(os, m_node_shape);
        m_tensor.DumpObject(os);
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
        bb::LoadValue(is, m_data_type);
        bb::LoadValue(is, m_frame_size);
        bb::LoadValue(is, m_frame_stride);
        bb::LoadValue(is, m_node_shape);
        m_tensor.LoadObject(is);
        
        // 再構築
        m_node_size = CalcShapeSize(m_node_shape);
    }


    // -------------------------------------
    //  Serialize(旧)
    // -------------------------------------
public:    
    void Save(std::ostream &os) const 
    {
        os.write((char const *)&m_data_type, sizeof(m_data_type));
        SaveIndex(os, m_frame_size);
        SaveIndex(os, m_frame_stride);
        SaveIndex(os, m_node_size);
        SaveIndices(os, m_node_shape);
        m_tensor.Save(os);
    }

    void Load(std::istream &is)
    {
        is.read((char *)&m_data_type, sizeof(m_data_type));
        m_frame_size   = LoadIndex(is); 
        m_frame_stride = LoadIndex(is);
        m_node_size    = LoadIndex(is);
        m_node_shape   = LoadIndices(is);
        m_tensor.Load(is);
    }

    void Save(std::string filename) const
    {
        std::ofstream ofs(filename, std::ios::binary);
        Save(ofs);
    }
    void Load(std::string filename)
    {
        std::ifstream ifs(filename, std::ios::binary);
        Load(ifs);
    }


#ifdef BB_WITH_CEREAL
    template <class Archive>
    void serialize(Archive& archive, std::uint32_t const version)
    {
        archive(cereal::make_nvp("data_type",    m_data_type));
        archive(cereal::make_nvp("frame_size",   m_frame_size));
        archive(cereal::make_nvp("frame_stride", m_frame_stride));
        archive(cereal::make_nvp("node_size",    m_node_size));
        archive(cereal::make_nvp("node_shape",   m_node_shape));
        archive(cereal::make_nvp("tensor",       m_tensor));
    }
#endif


    // -------------------------------------
    //  Resize
    // -------------------------------------
    
    /*
    void Resize(index_t frame_size, index_t i0, int type)                                      { Resize(frame_size, indices_t({i0}), type); }
    void Resize(index_t frame_size, index_t i0, index_t i1, int type)                          { Resize(frame_size, indices_t({i0, i1}), type); }
    void Resize(index_t frame_size, index_t i0, index_t i1, index_t i2, int type)              { Resize(frame_size, indices_t({i0, i1, i2}), type); }
    void Resize(index_t frame_size, index_t i0, index_t i1, index_t i2, index_t i3, int type)  { Resize(frame_size, indices_t({i0, i1, i2, i3}), type); }
    */

    void Reshape(std::vector<index_t> shape)
    {
        index_t auto_index = -1;
        index_t total = 1;
        for (index_t i = (index_t)shape.size()-1; i >= 0;--i) {
            if (shape[i] < 0) {
                auto_index = i;
            }
            else {
                total *= shape[i];
            }
        }
        if (auto_index >= 0) {
            shape[auto_index] = m_node_size / total;
        }

        // 再計算
        total = 1;
        for (auto size : shape) {
            total *= size;
        }
        BB_ASSERT(total == m_node_size);

        m_node_shape = shape;
        
        std::vector<index_t> tensor_shape;
        for ( auto size : shape ) {
            tensor_shape.push_back(size);
        }
        tensor_shape.push_back(-1);

        m_tensor.Reshape(tensor_shape);
    }

    std::vector<index_t> GetShape(void) const
    {
        return m_node_shape;
    }

    /**
     * @brief  内容のゼロ埋め
     * @detail 内容のゼロ埋め
     */
    void FillZero(void)
    {
        m_tensor.FillZero();
    }

//  void FillZeroMargin(void)
//  {
//  }
    

    int     GetType(void)  const { return m_data_type; }
    index_t GetFrameSize(void)  const { return m_frame_size; }
    index_t GetNodeSize(void)  const { return m_node_size; }


    // debug
    inline bool IsValidValue(void) const
    {
        return m_tensor.IsValidValue();
    }

    // debug
    template <typename Tp>
    inline bool IsZero(void) const
    {
        for (index_t frame = 0; frame < GetFrameSize(); ++frame ) {
            for (index_t node = 0; node < GetNodeSize(); ++node ) {
                if ( GetValue<Tp>(frame, node) != 0 ) {
                    return false;
                }
            }
        }
        return true;
    }


    // ---------------------------------
    //  ダイレクトアクセス用
    // ---------------------------------

    index_t GetFrameStride(void)  const { return m_frame_stride; }

    Memory::Ptr         LockMemory(bool new_buf=false) const    { return m_tensor.LockMemory(new_buf); }
    Memory::ConstPtr    LockMemoryConst(void) const             { return m_tensor.LockMemoryConst(); }
    Memory::DevPtr      LockDeviceMemory(bool new_buf=false) const { return m_tensor.LockDeviceMemory(new_buf); }
    Memory::DevConstPtr LockDeviceMemoryConst(void) const          { return m_tensor.LockDeviceMemoryConst(); }

    // 型指定アクセス
    template <typename MemTp, typename ValueTp>
    inline ValueTp Get(void const *addr, index_t frame, index_t node) const
    {
        BB_DEBUG_ASSERT(m_data_type == DataType<MemTp>::type);
        auto ptr = LockMemoryConst();
        return static_cast<ValueTp>(DataType_Read<MemTp>(GetNodeBaseAddr(addr, node), frame)); 
    }
    
    template <typename MemTp, typename ValueTp>
    inline ValueTp Get(void const *addr, index_t frame, std::vector<index_t> const & indices)
    {
        return Get<MemTp, ValueTp>(addr, frame, GetNodeIndex(indices));
    }

    template <typename MemTp, typename ValueTp>
    inline void Set(void *addr, index_t frame, index_t node, ValueTp value)
    {
        BB_DEBUG_ASSERT(m_data_type == DataType<MemTp>::type);
        auto ptr = LockMemory();
        DataType_Write<MemTp>(GetNodeBaseAddr(addr, node), frame, static_cast<MemTp>(value));
    }

    template <typename MemTp, typename ValueTp>
    inline void Set(void *addr, index_t frame, std::vector<index_t> const & indices, ValueTp value)
    {
        Set<MemTp, ValueTp>(addr, frame, GetNodeIndex(indices), value);
    }

    template <typename MemTp, typename ValueTp>
    inline void Add(void *addr, index_t frame, index_t node, ValueTp value)
    {
        BB_DEBUG_ASSERT(m_data_type == DataType<MemTp>::type);
        auto ptr = LockMemory();
        DataType_Add<MemTp>(GetNodeBaseAddr(addr, node), frame, static_cast<MemTp>(value));
    }

    template <typename MemTp, typename ValueTp>
    inline void Add(void *addr, index_t frame, std::vector<index_t> const & indices, ValueTp value)
    {
        Add<MemTp, ValueTp>(addr, frame, GetNodeIndex(indices), value);
    }


    // 汎用アクセス
    template <typename Tp>
    inline Tp GetValue(void const *addr, index_t frame, index_t node)
    {
        return ReadValue<Tp>(GetNodeBaseAddr(addr, node), frame);
    }

    template <typename Tp>
    inline Tp GetValue(void const *addr, index_t frame, std::vector<index_t> const & indices)
    {
        return GetValue<Tp>(addr, frame, GetNodeIndex(indices));
    }

    template <typename Tp>
    inline void SetValue(void *addr, index_t frame, index_t node, Tp value)
    {
        WriteValue<Tp>(GetNodeBaseAddr(addr, node), frame, value);
    }

    template <typename Tp>
    inline void SetValue(void *addr, index_t frame, std::vector<index_t> const & indices, Tp value)
    {
        SetValue<Tp>(addr, frame, GetNodeIndex(indices), value);
    }

    template <typename Tp>
    inline void AddValue(void *addr, index_t frame, index_t node, Tp value)
    {
        AddValue<Tp>(GetNodeBaseAddr(addr, node), frame, value);
    }

    template <typename Tp>
    inline void AddValue(void *addr, index_t frame, std::vector<index_t> const & indices, Tp value)
    {
        AddValue<Tp>(addr, frame, GetNodeIndex(indices), value);
    }


protected:
    template <typename DT, typename ST>
    FrameBuffer ConvertTo__(void)
    {
        BB_ASSERT(GetType() == DataType<ST>::type);

        FrameBuffer dst_buf(GetFrameSize(), GetShape(), DataType<DT>::type);

#ifdef BB_WITH_CUDA
        if ( DataType<ST>::type == BB_TYPE_BIT && DataType<DT>::type == BB_TYPE_FP32 && this->IsDeviceAvailable() && dst_buf.IsDeviceAvailable() ) {
            auto src_ptr = this->LockDeviceMemoryConst();
            auto dst_ptr = dst_buf.LockDeviceMemory(true);
            
            bbcu_ConvBitToReal<float>(
                    (int const *)src_ptr.GetAddr(),
                    (float     *)dst_ptr.GetAddr(),
                    0.0f,
                    1.0f,
                    (int)this->GetNodeSize(),
                    (int)this->GetFrameSize(),
                    (int)(this->GetFrameStride() / sizeof(int)),
                    (int)(dst_buf.GetFrameStride() / sizeof(float))
                );

            return dst_buf;
        }
#endif

        auto src_ptr = this->LockConst<ST>();
        auto dst_ptr = dst_buf.Lock<DT>();

        #pragma omp parallel for
        for (index_t node = 0; node < m_node_size; ++node) {
            for (index_t frame = 0; frame < m_frame_size; ++frame) {
                dst_ptr.Set(frame, node, (DT)src_ptr.Get(frame, node));
            }
        }

        return dst_buf;
    }

    template <typename ST>
    FrameBuffer ConvertTo_(int type)
    {
        switch (type) {
        case BB_TYPE_BIT:    return ConvertTo__<Bit,          ST>();
        case BB_TYPE_FP32:   return ConvertTo__<float,        ST>();
        case BB_TYPE_FP64:   return ConvertTo__<double,       ST>();
        case BB_TYPE_INT8:   return ConvertTo__<std::int8_t,  ST>();
        case BB_TYPE_INT16:  return ConvertTo__<std::int16_t, ST>();
        case BB_TYPE_INT32:  return ConvertTo__<std::int32_t, ST>();
        case BB_TYPE_INT64:  return ConvertTo__<std::int64_t, ST>();
        case BB_TYPE_UINT8:  return ConvertTo__<std::int8_t,  ST>();
        case BB_TYPE_UINT16: return ConvertTo__<std::int16_t, ST>();
        case BB_TYPE_UINT32: return ConvertTo__<std::int32_t, ST>();
        case BB_TYPE_UINT64: return ConvertTo__<std::int64_t, ST>();
        default:   BB_ASSERT(0);
        }
        return FrameBuffer();
    }

public:
    FrameBuffer ConvertTo(int type)
    {
        switch (m_data_type) {
        case BB_TYPE_BIT:    return ConvertTo_<Bit>(type);
        case BB_TYPE_FP32:   return ConvertTo_<float>(type);
        case BB_TYPE_FP64:   return ConvertTo_<double>(type);
        case BB_TYPE_INT8:   return ConvertTo_<std::int8_t >(type);
        case BB_TYPE_INT16:  return ConvertTo_<std::int16_t>(type);
        case BB_TYPE_INT32:  return ConvertTo_<std::int32_t>(type);
        case BB_TYPE_INT64:  return ConvertTo_<std::int64_t>(type);
        case BB_TYPE_UINT8:  return ConvertTo_<std::int8_t>(type);
        case BB_TYPE_UINT16: return ConvertTo_<std::int16_t>(type);
        case BB_TYPE_UINT32: return ConvertTo_<std::int32_t>(type);
        case BB_TYPE_UINT64: return ConvertTo_<std::int64_t>(type);
        default:   BB_ASSERT(0);
        }
        return FrameBuffer();
    }

protected:
    // ---------------------------------
    //  アクセス用
    // ---------------------------------

    void *GetNodeBaseAddr(void* base_addr, index_t node) const
    {
        BB_DEBUG_ASSERT(node < m_node_size);
        auto addr = (std::uint8_t*)base_addr;
        return addr + (m_frame_stride * node);
    }

    void const *GetNodeBaseAddr(const void* base_addr, index_t node) const
    {
        BB_DEBUG_ASSERT(node < m_node_size);
        auto addr = (std::uint8_t const *)base_addr;
        return addr + (m_frame_stride * node);
    }

    inline index_t GetNodeIndex(std::vector<index_t> const & indices) const
    {
        BB_DEBUG_ASSERT(indices.size() == m_node_shape.size());

        index_t stride = 1;
        index_t index = 0;
        for ( index_t i = (index_t)m_node_shape.size()-1; i>= 0; --i ) {
            BB_DEBUG_ASSERT(indices[i] >= 0 && indices[i] < m_node_shape[i]);
            index += stride * indices[i];
            stride *= m_node_shape[i];
        }
        return index;
    }

    template<typename Tp>
    Tp ReadValue(void *base, index_t frame) const
    {
        switch (m_data_type) {
        case BB_TYPE_BIT:    return static_cast<Tp>(DataType_Read<Bit>         (base, frame));  break;
        case BB_TYPE_FP32:   return static_cast<Tp>(DataType_Read<float>       (base, frame));  break;
        case BB_TYPE_FP64:   return static_cast<Tp>(DataType_Read<double>      (base, frame));  break;
        case BB_TYPE_INT8:   return static_cast<Tp>(DataType_Read<std::int8_t> (base, frame));  break;
        case BB_TYPE_INT16:  return static_cast<Tp>(DataType_Read<std::int16_t>(base, frame));  break;
        case BB_TYPE_INT32:  return static_cast<Tp>(DataType_Read<std::int32_t>(base, frame));  break;
        case BB_TYPE_INT64:  return static_cast<Tp>(DataType_Read<std::int64_t>(base, frame));  break;
        case BB_TYPE_UINT8:  return static_cast<Tp>(DataType_Read<std::int8_t> (base, frame));  break;
        case BB_TYPE_UINT16: return static_cast<Tp>(DataType_Read<std::int16_t>(base, frame));  break;
        case BB_TYPE_UINT32: return static_cast<Tp>(DataType_Read<std::int32_t>(base, frame));  break;
        case BB_TYPE_UINT64: return static_cast<Tp>(DataType_Read<std::int64_t>(base, frame));  break;
        default:   BB_ASSERT(0);
        }
        return 0;
    }

    template<typename Tp>
    void WriteValue(void *base, index_t frame, Tp value) const
    {
        switch (m_data_type) {
        case BB_TYPE_BIT:    DataType_Write<Bit>         (base, frame, static_cast<Bit>     (value));   break;
        case BB_TYPE_FP32:   DataType_Write<float>       (base, frame, static_cast<float>   (value));   break;
        case BB_TYPE_FP64:   DataType_Write<double>      (base, frame, static_cast<double>  (value));   break;
        case BB_TYPE_INT8:   DataType_Write<std::int8_t> (base, frame, static_cast<int8_t>  (value));   break;
        case BB_TYPE_INT16:  DataType_Write<std::int16_t>(base, frame, static_cast<int16_t> (value));   break;
        case BB_TYPE_INT32:  DataType_Write<std::int32_t>(base, frame, static_cast<int32_t> (value));   break;
        case BB_TYPE_INT64:  DataType_Write<std::int64_t>(base, frame, static_cast<int64_t> (value));   break;
        case BB_TYPE_UINT8:  DataType_Write<std::int8_t> (base, frame, static_cast<uint8_t> (value));   break;
        case BB_TYPE_UINT16: DataType_Write<std::int16_t>(base, frame, static_cast<uint16_t>(value));   break;
        case BB_TYPE_UINT32: DataType_Write<std::int32_t>(base, frame, static_cast<uint32_t>(value));   break;
        case BB_TYPE_UINT64: DataType_Write<std::int64_t>(base, frame, static_cast<uint64_t>(value));   break;
        default:   BB_ASSERT(0);
        }
    }

    template<typename Tp>
    void AddValue(void *base, index_t frame, Tp value) const
    {
        switch (m_data_type) {
        case BB_TYPE_BIT:    DataType_Add<Bit>         (base, frame, static_cast<Bit>     (value)); break;
        case BB_TYPE_FP32:   DataType_Add<float>       (base, frame, static_cast<float>   (value)); break;
        case BB_TYPE_FP64:   DataType_Add<double>      (base, frame, static_cast<double>  (value)); break;
        case BB_TYPE_INT8:   DataType_Add<std::int8_t> (base, frame, static_cast<int8_t>  (value)); break;
        case BB_TYPE_INT16:  DataType_Add<std::int16_t>(base, frame, static_cast<int16_t> (value)); break;
        case BB_TYPE_INT32:  DataType_Add<std::int32_t>(base, frame, static_cast<int32_t> (value)); break;
        case BB_TYPE_INT64:  DataType_Add<std::int64_t>(base, frame, static_cast<int64_t> (value)); break;
        case BB_TYPE_UINT8:  DataType_Add<std::int8_t> (base, frame, static_cast<uint8_t> (value)); break;
        case BB_TYPE_UINT16: DataType_Add<std::int16_t>(base, frame, static_cast<uint16_t>(value)); break;
        case BB_TYPE_UINT32: DataType_Add<std::int32_t>(base, frame, static_cast<uint32_t>(value)); break;
        case BB_TYPE_UINT64: DataType_Add<std::int64_t>(base, frame, static_cast<uint64_t>(value)); break;
        default:   BB_ASSERT(0);
        }
    }


public:

    friend FrameBufferConstPtr_<Bit      const, FrameBuffer const, Memory::ConstPtr>;
    friend FrameBufferConstPtr_<float    const, FrameBuffer const, Memory::ConstPtr>;
    friend FrameBufferConstPtr_<double   const, FrameBuffer const, Memory::ConstPtr>;
    friend FrameBufferConstPtr_<int8_t   const, FrameBuffer const, Memory::ConstPtr>;
    friend FrameBufferConstPtr_<int16_t  const, FrameBuffer const, Memory::ConstPtr>;
    friend FrameBufferConstPtr_<int32_t  const, FrameBuffer const, Memory::ConstPtr>;
    friend FrameBufferConstPtr_<int64_t  const, FrameBuffer const, Memory::ConstPtr>;
    friend FrameBufferConstPtr_<uint8_t  const, FrameBuffer const, Memory::ConstPtr>;
    friend FrameBufferConstPtr_<uint16_t const, FrameBuffer const, Memory::ConstPtr>;
    friend FrameBufferConstPtr_<uint32_t const, FrameBuffer const, Memory::ConstPtr>;
    friend FrameBufferConstPtr_<uint64_t const, FrameBuffer const, Memory::ConstPtr>;

    friend FrameBufferConstPtr_<Bit     , FrameBuffer, Memory::Ptr>;
    friend FrameBufferConstPtr_<float   , FrameBuffer, Memory::Ptr>;
    friend FrameBufferConstPtr_<double  , FrameBuffer, Memory::Ptr>;
    friend FrameBufferConstPtr_<int8_t  , FrameBuffer, Memory::Ptr>;
    friend FrameBufferConstPtr_<int16_t , FrameBuffer, Memory::Ptr>;
    friend FrameBufferConstPtr_<int32_t , FrameBuffer, Memory::Ptr>;
    friend FrameBufferConstPtr_<int64_t , FrameBuffer, Memory::Ptr>;
    friend FrameBufferConstPtr_<uint8_t , FrameBuffer, Memory::Ptr>;
    friend FrameBufferConstPtr_<uint16_t, FrameBuffer, Memory::Ptr>;
    friend FrameBufferConstPtr_<uint32_t, FrameBuffer, Memory::Ptr>;
    friend FrameBufferConstPtr_<uint64_t, FrameBuffer, Memory::Ptr>;

    friend FrameBufferPtr_<Bit     , FrameBuffer, Memory::Ptr>;
    friend FrameBufferPtr_<float   , FrameBuffer, Memory::Ptr>;
    friend FrameBufferPtr_<double  , FrameBuffer, Memory::Ptr>;
    friend FrameBufferPtr_<int8_t  , FrameBuffer, Memory::Ptr>;
    friend FrameBufferPtr_<int16_t , FrameBuffer, Memory::Ptr>;
    friend FrameBufferPtr_<int32_t , FrameBuffer, Memory::Ptr>;
    friend FrameBufferPtr_<int64_t , FrameBuffer, Memory::Ptr>;
    friend FrameBufferPtr_<uint8_t , FrameBuffer, Memory::Ptr>;
    friend FrameBufferPtr_<uint16_t, FrameBuffer, Memory::Ptr>;
    friend FrameBufferPtr_<uint32_t, FrameBuffer, Memory::Ptr>;
    friend FrameBufferPtr_<uint64_t, FrameBuffer, Memory::Ptr>;

    template <typename Tp>
    auto LockConst(void) const
    {
        FrameBufferConstPtr_<Tp const, FrameBuffer const, Memory::ConstPtr> ptr(this);
        ptr.Lock();
        return ptr;
    }

    template <typename Tp>
    auto Lock(bool new_buf=false)
    {
        FrameBufferPtr_<Tp, FrameBuffer, Memory::Ptr> ptr(this);
        ptr.Lock(new_buf);
        return ptr;
    }


public:

    // 型指定アクセス
    template <typename MemTp, typename ValueTp>
    inline ValueTp Get(index_t frame, index_t node) const
    {
        BB_DEBUG_ASSERT(m_data_type == DataType<MemTp>::type);
        auto ptr = LockMemoryConst();
        return static_cast<ValueTp>(DataType_Read<MemTp>(GetNodeBaseAddr(ptr.GetAddr(), node), frame)); 
    }
    
    template <typename MemTp, typename ValueTp>
    inline ValueTp Get(index_t frame, std::vector<index_t> const & indices)
    {
        return Get<MemTp, ValueTp>(frame, GetNodeIndex(indices));
    }

    template <typename MemTp, typename ValueTp>
    inline void Set(index_t frame, index_t node, ValueTp value)
    {
        BB_DEBUG_ASSERT(m_data_type == DataType<MemTp>::type);
        auto ptr = LockMemory();
        DataType_Write<MemTp>(GetNodeBaseAddr(ptr.GetAddr(), node), frame, static_cast<MemTp>(value));
    }

    template <typename MemTp, typename ValueTp>
    inline void Set(index_t frame, std::vector<index_t> const & indices, ValueTp value)
    {
        Set<MemTp, ValueTp>(frame, GetNodeIndex(indices), value);
    }

    template <typename MemTp, typename ValueTp>
    inline void Add(index_t frame, index_t node, ValueTp value)
    {
        BB_DEBUG_ASSERT(m_data_type == DataType<MemTp>::type);
        auto ptr = LockMemory();
        DataType_Add<MemTp>(GetNodeBaseAddr(ptr.GetAddr(), node), frame, static_cast<MemTp>(value));
    }

    template <typename MemTp, typename ValueTp>
    inline void Add(index_t frame, std::vector<index_t> const & indices, ValueTp value)
    {
        Add<MemTp, ValueTp>(frame, GetNodeIndex(indices), value);
    }


    // 汎用アクセス
    template <typename Tp>
    inline Tp GetValue(index_t frame, index_t node) const
    {
        auto ptr = LockMemory();
        return ReadValue<Tp>(GetNodeBaseAddr(ptr.GetAddr(), node), frame);
    }

    template <typename Tp>
    inline Tp GetValue(index_t frame, std::vector<index_t> const & indices) const
    {
        return GetValue<Tp>(frame, GetNodeIndex(indices));
    }

    template <typename Tp>
    inline void SetValue(index_t frame, index_t node, Tp value)
    {
        auto ptr = LockMemory();
        WriteValue<Tp>(GetNodeBaseAddr(ptr.GetAddr(), node), frame, value);
    }

    template <typename Tp>
    inline void SetValue(index_t frame, std::vector<index_t> const & indices, Tp value)
    {
        SetValue<Tp>(frame, GetNodeIndex(indices), value);
    }

    template <typename Tp>
    inline void AddValue(index_t frame, index_t node, Tp value)
    {
        auto ptr = LockMemory();
        AddValue<Tp>(GetNodeBaseAddr(ptr.GetAddr(), node), frame, value);
    }

    template <typename Tp>
    inline void AddValue(index_t frame, std::vector<index_t> const & indices, Tp value)
    {
        AddValue<Tp>(frame, GetNodeIndex(indices), value);
    }


    inline void SetBit   (index_t frame, index_t node, Bit           value) { SetValue<Bit          >(frame, node, value); }
    inline void SetFP32  (index_t frame, index_t node, float         value) { SetValue<float        >(frame, node, value); }
    inline void SetFP64  (index_t frame, index_t node, double        value) { SetValue<double       >(frame, node, value); }
    inline void SetINT8  (index_t frame, index_t node, std::int8_t   value) { SetValue<std::int8_t  >(frame, node, value); }
    inline void SetINT16 (index_t frame, index_t node, std::int16_t  value) { SetValue<std::int16_t >(frame, node, value); }
    inline void SetINT32 (index_t frame, index_t node, std::int32_t  value) { SetValue<std::int32_t >(frame, node, value); }
    inline void SetINT64 (index_t frame, index_t node, std::int64_t  value) { SetValue<std::int64_t >(frame, node, value); }
    inline void SetUINT8 (index_t frame, index_t node, std::uint8_t  value) { SetValue<std::uint8_t >(frame, node, value); }
    inline void SetUINT16(index_t frame, index_t node, std::uint16_t value) { SetValue<std::uint16_t>(frame, node, value); }
    inline void SetUINT32(index_t frame, index_t node, std::uint32_t value) { SetValue<std::uint32_t>(frame, node, value); }
    inline void SetUINT64(index_t frame, index_t node, std::uint64_t value) { SetValue<std::uint64_t>(frame, node, value); }

    inline void SetBit   (index_t frame, std::vector<index_t> const & indices, Bit           value) { SetValue<Bit          >(frame, indices, value); }
    inline void SetFP32  (index_t frame, std::vector<index_t> const & indices, float         value) { SetValue<float        >(frame, indices, value); }
    inline void SetFP64  (index_t frame, std::vector<index_t> const & indices, double        value) { SetValue<double       >(frame, indices, value); }
    inline void SetINT8  (index_t frame, std::vector<index_t> const & indices, std::int8_t   value) { SetValue<std::int8_t  >(frame, indices, value); }
    inline void SetINT16 (index_t frame, std::vector<index_t> const & indices, std::int16_t  value) { SetValue<std::int16_t >(frame, indices, value); }
    inline void SetINT32 (index_t frame, std::vector<index_t> const & indices, std::int32_t  value) { SetValue<std::int32_t >(frame, indices, value); }
    inline void SetINT64 (index_t frame, std::vector<index_t> const & indices, std::int64_t  value) { SetValue<std::int64_t >(frame, indices, value); }
    inline void SetUINT8 (index_t frame, std::vector<index_t> const & indices, std::uint8_t  value) { SetValue<std::uint8_t >(frame, indices, value); }
    inline void SetUINT16(index_t frame, std::vector<index_t> const & indices, std::uint16_t value) { SetValue<std::uint16_t>(frame, indices, value); }
    inline void SetUINT32(index_t frame, std::vector<index_t> const & indices, std::uint32_t value) { SetValue<std::uint32_t>(frame, indices, value); }
    inline void SetUINT64(index_t frame, std::vector<index_t> const & indices, std::uint64_t value) { SetValue<std::uint64_t>(frame, indices, value); }

    inline Bit           GetBit   (index_t frame, index_t node) const { return GetValue<Bit          >(frame, node); }
    inline float         GetFP32  (index_t frame, index_t node) const { return GetValue<float        >(frame, node); }
    inline double        GetFP64  (index_t frame, index_t node) const { return GetValue<double       >(frame, node); }
    inline std::int8_t   GetINT8  (index_t frame, index_t node) const { return GetValue<std::int8_t  >(frame, node); }
    inline std::int16_t  GetINT16 (index_t frame, index_t node) const { return GetValue<std::int16_t >(frame, node); }
    inline std::int32_t  GetINT32 (index_t frame, index_t node) const { return GetValue<std::int32_t >(frame, node); }
    inline std::int64_t  GetINT64 (index_t frame, index_t node) const { return GetValue<std::int64_t >(frame, node); }
    inline std::uint8_t  GetUINT8 (index_t frame, index_t node) const { return GetValue<std::uint8_t >(frame, node); }
    inline std::uint16_t GetUINT16(index_t frame, index_t node) const { return GetValue<std::uint16_t>(frame, node); }
    inline std::uint32_t GetUINT32(index_t frame, index_t node) const { return GetValue<std::uint32_t>(frame, node); }
    inline std::uint64_t GetUINT64(index_t frame, index_t node) const { return GetValue<std::uint64_t>(frame, node); }
    
    inline Bit           GetBit   (index_t frame, std::vector<index_t> const & indices) { return GetValue<Bit          >(frame, indices); }
    inline float         GetFP32  (index_t frame, std::vector<index_t> const & indices) { return GetValue<float        >(frame, indices); }
    inline double        GetFP64  (index_t frame, std::vector<index_t> const & indices) { return GetValue<double       >(frame, indices); }
    inline std::int8_t   GetINT8  (index_t frame, std::vector<index_t> const & indices) { return GetValue<std::int8_t  >(frame, indices); }
    inline std::int16_t  GetINT16 (index_t frame, std::vector<index_t> const & indices) { return GetValue<std::int16_t >(frame, indices); }
    inline std::int32_t  GetINT32 (index_t frame, std::vector<index_t> const & indices) { return GetValue<std::int32_t >(frame, indices); }
    inline std::int64_t  GetINT64 (index_t frame, std::vector<index_t> const & indices) { return GetValue<std::int64_t >(frame, indices); }
    inline std::uint8_t  GetUINT8 (index_t frame, std::vector<index_t> const & indices) { return GetValue<std::uint8_t >(frame, indices); }
    inline std::uint16_t GetUINT16(index_t frame, std::vector<index_t> const & indices) { return GetValue<std::uint16_t>(frame, indices); }
    inline std::uint32_t GetUINT32(index_t frame, std::vector<index_t> const & indices) { return GetValue<std::uint32_t>(frame, indices); }
    inline std::uint64_t GetUINT64(index_t frame, std::vector<index_t> const & indices) { return GetValue<std::uint64_t>(frame, indices); }

    
    template<typename Tp>
    void SetVector(index_t frame, std::vector<Tp> const &data)
    {
        BB_ASSERT(data.size() == (size_t)m_node_size);
        BB_ASSERT(frame >= 0 && frame < m_frame_size);

        for (index_t node = 0; node < m_node_size; ++node) {
            SetValue<Tp>(frame, node, data[node]);
        }
    }

    template<typename Tp>
    void SetVector(std::vector< std::vector<Tp> > const &data)
    {
        BB_ASSERT(data.size() == (size_t)m_frame_size);
        for (index_t frame = 0; frame < m_frame_size; ++frame) {
            BB_ASSERT(data[frame].size() == (size_t)m_node_size);
            for (index_t node = 0; node < m_node_size; ++node) {
                SetValue<Tp>(frame, node, data[frame][node]);
            }
        }
    }

    template<typename Tp>
    void SetVector(std::vector< std::vector<Tp> > const &data, index_t offset)
    {
        BB_ASSERT(GetType() == DataType<Tp>::type);
        BB_ASSERT(offset + m_frame_size <= (index_t)data.size() );

        auto ptr = Lock<Tp>();
        for (index_t frame = 0; frame < m_frame_size; ++frame) {
            BB_ASSERT(data[frame].size() == (size_t)m_node_size);
            for (index_t node = 0; node < m_node_size; ++node) {
                ptr.Set(frame, node, data[frame + offset][node]);
            }
        }
    }

    template<typename Tp>
    std::vector<Tp> GetVector(index_t frame) const
    {
        BB_ASSERT(frame >= 0 && frame < m_frame_size);

        std::vector<Tp> data(m_node_size);
        for (index_t node = 0; node < m_node_size; ++node) {
            data[node] = GetValue<Tp>(frame, node);
        }

        return data;
    }


    template<typename BufType, typename VecType=float>
    void SetData_(std::vector< std::vector<VecType> > const &data, index_t offset=0)
    {
        BB_ASSERT(GetType() == DataType<BufType>::type);
        BB_ASSERT(offset + (index_t)data.size() <= m_frame_size);

        index_t size = (index_t)data.size(); 
        if ( size + offset > m_frame_size ) { size = m_frame_size - offset; }

        auto ptr = Lock<BufType>();
        for (index_t i = 0; i < size; ++i) {
            index_t frame = i + offset;
            BB_ASSERT(data[i].size() == (size_t)m_node_size);
            for (index_t node = 0; node < m_node_size; ++node) {
                ptr.Set(frame, node, (BufType)data[i][node]);
            }
        }
    }

    template<typename BufType, typename VecType=float>
    std::vector< std::vector<VecType> > GetData_(index_t size=0, index_t offset=0)
    {
        if ( size <= 0 ) { size = m_frame_size - offset; }

        BB_ASSERT(GetType() == DataType<BufType>::type);
        BB_ASSERT(offset + size <= m_frame_size);

        std::vector< std::vector<VecType> > data(size);

        auto ptr = LockConst<BufType>();
        for (index_t i = 0; i < size; ++i) {
            data[i].resize(m_node_size);
            index_t frame = i + offset;
            for (index_t node = 0; node < m_node_size; ++node) {
                data[i][node] = (VecType)ptr.Get(frame, node);
            }
        }

        return data;
    }
    
    template<typename VecType=float>
    void SetData(std::vector< std::vector<VecType> > const &data, index_t offset=0)
    {
        switch (GetType()) {
        case BB_TYPE_BIT:    SetData_<bb::Bit,       VecType>(data, offset);    break;
        case BB_TYPE_FP32:   SetData_<float,         VecType>(data, offset);    break;
        case BB_TYPE_FP64:   SetData_<double,        VecType>(data, offset);    break;
        case BB_TYPE_INT8:   SetData_<std::int8_t,   VecType>(data, offset);    break;
        case BB_TYPE_INT16:  SetData_<std::int16_t,  VecType>(data, offset);    break;
        case BB_TYPE_INT32:  SetData_<std::int32_t,  VecType>(data, offset);    break;
        case BB_TYPE_INT64:  SetData_<std::int64_t,  VecType>(data, offset);    break;
        case BB_TYPE_UINT8:  SetData_<std::uint8_t,  VecType>(data, offset);    break;
        case BB_TYPE_UINT16: SetData_<std::uint16_t, VecType>(data, offset);    break;
        case BB_TYPE_UINT32: SetData_<std::uint32_t, VecType>(data, offset);    break;
        case BB_TYPE_UINT64: SetData_<std::uint64_t, VecType>(data, offset);    break;
        default:   BB_ASSERT(0);
        }
    }

    template<typename VecType=float>
    std::vector< std::vector<VecType> > GetData(index_t size=0, index_t offset=0)
    {
        switch (GetType()) {
        case BB_TYPE_BIT:    return GetData_<bb::Bit,       VecType>(size, offset);
        case BB_TYPE_FP32:   return GetData_<float,         VecType>(size, offset);
        case BB_TYPE_FP64:   return GetData_<double,        VecType>(size, offset);
        case BB_TYPE_INT8:   return GetData_<std::int8_t,   VecType>(size, offset);
        case BB_TYPE_INT16:  return GetData_<std::int16_t,  VecType>(size, offset);
        case BB_TYPE_INT32:  return GetData_<std::int32_t,  VecType>(size, offset);
        case BB_TYPE_INT64:  return GetData_<std::int64_t,  VecType>(size, offset);
        case BB_TYPE_UINT8:  return GetData_<std::uint8_t,  VecType>(size, offset);
        case BB_TYPE_UINT16: return GetData_<std::uint16_t, VecType>(size, offset);
        case BB_TYPE_UINT32: return GetData_<std::uint32_t, VecType>(size, offset);
        case BB_TYPE_UINT64: return GetData_<std::uint64_t, VecType>(size, offset);
        default:   BB_ASSERT(0);
        }
        return std::vector< std::vector<VecType> >();
    }


    // テンソルの設定
public:
    /*
    template<typename Tp>
    void SetTensor(index_t frame, Tensor const &tensor)
    {
        BB_ASSERT(m_data_type == DataType<Tp>::type);
        BB_ASSERT(tensor.GetType() == DataType<Tp>::type);
        BB_ASSERT(tensor.GetShape() == m_tensor.GetShape());
        
        int dim = tensor.GetDim();
        indices_t indices(tensor.GetDim(), 0);
        for ( ; ; ) {
//            m_tensor.template At<Tp>(indices) = tensor.template At<Tp>(indices);

            int i = 0;
            for ( ; ; ) {
                indices[i]++;
                if ( indices[i] < m_node_shape[i] ) {
                    break;
                }
                indices[i] = 0;
                i++;
                if (i >= dim) {
                    return;
                }
            }
        }
    }
    */

    // フレームの部分切り出し
    FrameBuffer FrameRange(index_t size, index_t offset=0)
    {
        BB_ASSERT(offset >= 0 && offset < m_frame_size);
        BB_ASSERT(size >= 0 &&  size < m_frame_size - offset);

        FrameBuffer buf(size, m_node_shape, m_data_type);

        auto src_ptr = m_tensor.LockMemoryConst();
        auto dst_ptr = buf.m_tensor.LockMemory(true);
        auto src_addr = (std::int8_t const *)src_ptr.GetAddr();
        auto dst_addr = (std::int8_t       *)dst_ptr.GetAddr();

        if (m_data_type == BB_TYPE_BIT && (offset % 8) != 0 ) {
            #pragma omp parallel for
            for (index_t node = 0; node < m_node_size; ++node)
            {
                for (index_t frame = 0; frame < size; ++frame) {
                    auto val = DataType_Read<Bit>(src_addr + m_frame_stride * node, frame + offset);
                    DataType_Write<Bit>(dst_addr + m_frame_stride * node, frame, val);
                }
            }           
        }
        else {
            int     unit   = DataType_GetBitSize(m_data_type);
            index_t byte_offset = (offset * unit + 7) / 8;
            index_t byte_size   = (size * unit + 7) / 8;

            #pragma omp parallel for
            for (index_t node = 0; node < m_node_size; ++node)
            {
                memcpy(dst_addr + buf.m_frame_stride * node, src_addr + m_frame_stride * node + byte_offset, byte_size);
            }
        }

        return buf;
    }

    
    FrameBuffer Range(index_t size, index_t offset=0)
    {
        BB_ASSERT(offset >= 0 && offset < m_node_size);
        BB_ASSERT(size >= 0 &&  size < m_node_size - offset);

        FrameBuffer buf(m_frame_size, {size}, m_data_type);

        auto src_ptr = m_tensor.LockMemoryConst();
        auto dst_ptr = buf.m_tensor.LockMemory(true);
        auto src_addr = (std::int8_t const *)src_ptr.GetAddr();
        auto dst_addr = (std::int8_t       *)dst_ptr.GetAddr();

        index_t stride_size = GetFrameStride();
        memcpy(dst_addr, src_addr + (m_frame_stride * offset), stride_size * size);

        return buf;
    }


    FrameBuffer Concatenate(FrameBuffer const& buf)
    {
        BB_ASSERT(buf.GetType() == GetType());
        BB_ASSERT(buf.m_frame_size == m_frame_size);
        BB_ASSERT(buf.m_frame_stride == m_frame_stride);

        FrameBuffer dst_buf(m_frame_size, {m_node_size + buf.m_node_size}, m_data_type);

        auto dst_ptr  = dst_buf.m_tensor.LockMemory(true);
        auto dst_addr = (std::int8_t *)dst_ptr.GetAddr();

        {
            auto src0_ptr  = m_tensor.LockMemoryConst();
            auto src0_addr = (std::int8_t const *)src0_ptr.GetAddr();
            memcpy(dst_addr, src0_addr, m_frame_stride * m_node_size);
            dst_addr += m_frame_stride * m_node_size;
        }

        {
            auto src1_ptr  = buf.m_tensor.LockMemoryConst();
            auto src1_addr = (std::int8_t const *)src1_ptr.GetAddr();
            memcpy(dst_addr, src1_addr, buf.m_frame_stride * buf.m_node_size);
        }

        return dst_buf;
    }



    // -------------------------------------
    //  演算
    // -------------------------------------

    inline FrameBuffer& operator+=(FrameBuffer src) { m_tensor += src.m_tensor; return *this; }
    inline FrameBuffer& operator+=(double src)      { m_tensor += src; return *this; }
    inline FrameBuffer& operator-=(FrameBuffer src) { m_tensor -= src.m_tensor; return *this; }
    inline FrameBuffer& operator-=(double src)      { m_tensor -= src; return *this; }
    inline FrameBuffer& operator*=(FrameBuffer src) { m_tensor *= src.m_tensor; return *this; }
    inline FrameBuffer& operator*=(double src)      { m_tensor *= src; return *this; }
    inline FrameBuffer& operator/=(FrameBuffer src) { m_tensor /= src.m_tensor; return *this; }
    inline FrameBuffer& operator/=(double src)      { m_tensor /= src; return *this; }

    
    FrameBuffer Sqrt(void)
    {
        FrameBuffer dst(GetFrameSize(), GetShape(), GetType(), IsHostOnly());
        dst.m_tensor = m_tensor.Sqrt();
        return dst;
    }

    FrameBuffer Exp(void)
    {
        FrameBuffer dst(GetFrameSize(), GetShape(), GetType(), IsHostOnly());
        dst.m_tensor = m_tensor.Exp();
        return dst;
    }
    

    double Sum(void)
    {
        return m_tensor.Sum();
    }

    double Norm(void)
    {
        return std::sqrt((*this * *this).Sum());
    }



    friend  FrameBuffer operator+(FrameBuffer const &src0, FrameBuffer const &src1);
    friend  FrameBuffer operator+(FrameBuffer const &src0, double src1);
    friend  FrameBuffer operator+(double src0, FrameBuffer const &src1);
    friend  FrameBuffer operator-(FrameBuffer const &src0, FrameBuffer const &src1);
    friend  FrameBuffer operator-(FrameBuffer const &src0, double src1);
    friend  FrameBuffer operator-(double src0, FrameBuffer const &src1);
    friend  FrameBuffer operator*(FrameBuffer const &src0, FrameBuffer const &src1);
    friend  FrameBuffer operator*(FrameBuffer const &src0, double src1);
    friend  FrameBuffer operator*(double src0, FrameBuffer const &src1);
    friend  FrameBuffer operator/(FrameBuffer const &src0, FrameBuffer const &src1);
    friend  FrameBuffer operator/(FrameBuffer const &src0, double src1);
    friend  FrameBuffer operator/(double src0, FrameBuffer const &src1);
    friend  FrameBuffer Sqrt(FrameBuffer const &src);
    friend  FrameBuffer Exp(FrameBuffer const &src);
};


inline FrameBuffer operator+(FrameBuffer const &src0, FrameBuffer const &src1)
{
    FrameBuffer dst(src0.GetFrameSize(), src0.GetShape(), src0.GetType(), src0.IsHostOnly());
    dst.m_tensor = src0.m_tensor + src1.m_tensor;
    return dst;
}

inline FrameBuffer operator+(FrameBuffer const &src0, double src1)
{
    FrameBuffer dst(src0.GetFrameSize(), src0.GetShape(), src0.GetType(), src0.IsHostOnly());
    dst.m_tensor = src0.m_tensor + src1;
    return dst;
}

inline FrameBuffer operator+(double src0, FrameBuffer const &src1)
{
    FrameBuffer dst(src1.GetFrameSize(), src1.GetShape(), src1.GetType(), src1.IsHostOnly());
    dst.m_tensor = src0 + src1.m_tensor;
    return dst;
}


inline FrameBuffer operator-(FrameBuffer const &src0, FrameBuffer const &src1)
{
    FrameBuffer dst(src0.GetFrameSize(), src0.GetShape(), src0.GetType(), src0.IsHostOnly());
    dst.m_tensor = src0.m_tensor - src1.m_tensor;
    return dst;
}

inline FrameBuffer operator-(FrameBuffer const &src0, double src1)
{
    FrameBuffer dst(src0.GetFrameSize(), src0.GetShape(), src0.GetType(), src0.IsHostOnly());
    dst.m_tensor = src0.m_tensor - src1;
    return dst;
}

inline FrameBuffer operator-(double src0, FrameBuffer const &src1)
{
    FrameBuffer dst(src1.GetFrameSize(), src1.GetShape(), src1.GetType(), src1.IsHostOnly());
    dst.m_tensor = src0 - src1.m_tensor;
    return dst;
}


inline FrameBuffer operator*(FrameBuffer const &src0, FrameBuffer const &src1)
{
    FrameBuffer dst(src0.GetFrameSize(), src0.GetShape(), src0.GetType(), src0.IsHostOnly());
    dst.m_tensor = src0.m_tensor * src1.m_tensor;
    return dst;
}

inline FrameBuffer operator*(FrameBuffer const &src0, double src1)
{
    FrameBuffer dst(src0.GetFrameSize(), src0.GetShape(), src0.GetType(), src0.IsHostOnly());
    dst.m_tensor = src0.m_tensor * src1;
    return dst;
}

inline FrameBuffer operator*(double src0, FrameBuffer const &src1)
{
    FrameBuffer dst(src1.GetFrameSize(), src1.GetShape(), src1.GetType(), src1.IsHostOnly());
    dst.m_tensor = src0 * src1.m_tensor;
    return dst;
}


inline FrameBuffer operator/(FrameBuffer const &src0, FrameBuffer const &src1)
{
    FrameBuffer dst(src0.GetFrameSize(), src0.GetShape(), src0.GetType(), src0.IsHostOnly());
    dst.m_tensor = src0.m_tensor / src1.m_tensor;
    return dst;
}

inline FrameBuffer operator/(FrameBuffer const &src0, double src1)
{
    FrameBuffer dst(src0.GetFrameSize(), src0.GetShape(), src0.GetType(), src0.IsHostOnly());
    dst.m_tensor = src0.m_tensor / src1;
    return dst;
}

inline FrameBuffer operator/(double src0, FrameBuffer const &src1)
{
    FrameBuffer dst(src1.GetFrameSize(), src1.GetShape(), src1.GetType(), src1.IsHostOnly());
    dst.m_tensor = src0 / src1.m_tensor;
    return dst;
}

inline FrameBuffer Sqrt(FrameBuffer const &src)
{
    FrameBuffer dst(src.GetFrameSize(), src.GetShape(), src.GetType(), src.IsHostOnly());
    dst.m_tensor = Sqrt(src.m_tensor);
    return dst;
}

inline FrameBuffer Exp(FrameBuffer const &src)
{
    FrameBuffer dst(src.GetFrameSize(), src.GetShape(), src.GetType(), src.IsHostOnly());
    dst.m_tensor = Exp(src.m_tensor);
    return dst;
}


template<typename T>
std::ostream &operator<<(std::ostream& os, FrameBufferConstPtr_<T const, FrameBuffer const, Memory::ConstPtr> ptr)
{
    os << "[\n";
    for ( index_t frame = 0; frame < ptr.GetFrameBuffer().GetFrameSize(); ++frame ) {
        os << " [";
        for ( index_t node = 0; node < ptr.GetFrameBuffer().GetNodeSize(); ++node ) {
            os << ptr.Get(frame, node) << ", ";
            if ( node % 16 == 15 ) { os << "\n"; }
        }
        os << "]\n";
    }
    os << "]\n";
    return os;
}


inline std::ostream& operator<<(std::ostream& os, FrameBuffer const &buf)
{
    switch (buf.GetType()) {
    case BB_TYPE_BIT:    return os << buf.LockConst<Bit     >();
    case BB_TYPE_FP32:   return os << buf.LockConst<float   >();
    case BB_TYPE_FP64:   return os << buf.LockConst<double  >();
    case BB_TYPE_INT8:   return os << buf.LockConst<int8_t  >();
    case BB_TYPE_INT16:  return os << buf.LockConst<int16_t >();
    case BB_TYPE_INT32:  return os << buf.LockConst<int32_t >();
    case BB_TYPE_INT64:  return os << buf.LockConst<int64_t >();
    case BB_TYPE_UINT8:  return os << buf.LockConst<uint8_t >();
    case BB_TYPE_UINT16: return os << buf.LockConst<uint16_t>();
    case BB_TYPE_UINT32: return os << buf.LockConst<uint32_t>();
    case BB_TYPE_UINT64: return os << buf.LockConst<uint64_t>();
    default:   BB_ASSERT(0);
    }
    return os;
}


}
