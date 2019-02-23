// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once

//#include <stdlib.h>
//#include <malloc.h>

#include <mutex>
#include <iostream>
#include <array>
#include <vector>
#include <memory>
#include <malloc.h>

#include "bb/DataType.h"
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


#if 1

// -------------------------------------
//  アクセス用ポインタクラス定義
// -------------------------------------

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
        m_ptr = m_buf->GetMemoryConstPtr();
    }

    inline void const *GetNodeBaseAddr(index_t node) const
    {
        auto addr = (std::uint8_t const *)m_ptr.GetAddr();
        return addr + (m_buf->m_frame_stride * node);
    }

    inline index_t GetNodeIndex(indices_t const & indices) const
    {
        return GetShapeIndex(indices, m_buf->m_shape);
    }

  	inline Tp ReadValue(void const *base, index_t frame) const
	{
        return DataType_Read<Tp>(base, frame);
	}


public:
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
        return ReadValue(frame, GetNodeIndex(indices));
    }

    inline Tp Get(index_t frame, index_t i1, index_t i0) const
    {
        return ReadValue(frame, indices_t{i0, i1});
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
        m_ptr = m_buf->GetMemoryPtr(new_buf);
    }

protected:
    inline void *GetNodeBaseAddr(index_t node)
    {
        auto addr = (std::uint8_t *)m_ptr.GetAddr();
        return addr + (m_buf->m_frame_stride * node);
    }

    inline index_t GetNodeIndex(indices_t const & indices)
    {
        return GetShapeIndex(indices, m_buf->m_shape);
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

    inline void Set(index_t frame, index_t i1, index_t i0, Tp value)
    {
        return Set(frame, indices_t{i0, i1}, value);
    }


    inline void Add(index_t frame, index_t node, Tp value)
    {
        return AddValue(GetNodeBaseAddr(node),  frame, value);
    }

    inline void Add(index_t frame, indices_t indices, Tp value)
    {
        return Add(frame, GetNodeIndex(indices), value);
    }

    inline void Add(index_t frame, index_t i1, index_t i0, Tp value)
    {
        return Add(frame, indices_t{i0, i1}, value);
    }
};
#endif



// NeuralNet用のバッファ
class FrameBuffer
{
protected:
    Tensor                  m_tensor;

	int		                m_data_type = 0;
	index_t	                m_frame_size = 0;
	index_t                 m_frame_stride = 0;
	index_t	                m_node_size = 0;
    std::vector<index_t>    m_node_shape;

public:
   	/**
      * @brief  デフォルトコンストラクタ
      * @detail デフォルトコンストラクタ
      */
	FrameBuffer(bool hostOnly=false) : m_tensor(hostOnly) {}

  	/**
     * @brief  コンストラクタ
     * @detail コンストラクタ
     *        tensor は node_size サイズの1次元で初期化
     * @param frame_size フレーム数
     * @param node_size  1フレームのノード数
	 * @param data_type  1ノードのデータ型
     */
	FrameBuffer(int data_type, index_t frame_size, index_t node_size, bool hostOnly=false) : m_tensor(hostOnly)
	{
        Resize(data_type, frame_size, {node_size});
	}

   	/**
     * @brief  コンストラクタ
     * @detail コンストラクタ
     * @param frame_size フレーム数
     * @param shape      1フレームのノードを構成するshape
	 * @param data_type  1ノードのデータ型
     */
	FrameBuffer(int data_type, index_t frame_size, std::vector<index_t> shape, bool hostOnly=false) : m_tensor(hostOnly)
	{
		Resize(data_type, frame_size, shape);
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
    void Resize(int data_type, index_t frame_size, indices_t shape)
	{
        m_data_type    = data_type;
        m_frame_size   = frame_size;
        m_frame_stride = ((frame_size * DataType_GetBitSize(data_type) + 255) / 256) * (256 / 8);        // frame軸は256bit境界にあわせる(SIMD命令用)
        m_node_shape   = shape;


        // Bit型は内部 UINT8 で扱う
        int tensor_type = data_type;
        if ( data_type == BB_TYPE_BIT )
        {
            tensor_type = BB_TYPE_UINT8;
        }

        // サイズ計算
		m_node_size = 1;
        std::vector<index_t>    tensor_shape;
        tensor_shape.push_back(m_frame_stride / DataType_GetByteSize(tensor_type));
        for ( auto size : shape ) {
            tensor_shape.push_back(size);
    		m_node_size *= size;
        }

		// メモリ確保
		m_tensor.Resize(tensor_type, tensor_shape);
	}

    void Resize(int type, index_t frame_size, index_t i0)                                      { Resize(type, frame_size, indices_t({i0})); }
    void Resize(int type, index_t frame_size, index_t i0, index_t i1)                          { Resize(type, frame_size, indices_t({i0, i1})); }
    void Resize(int type, index_t frame_size, index_t i0, index_t i1, index_t i2)              { Resize(type, frame_size, indices_t({i0, i1, i2})); }
    void Resize(int type, index_t frame_size, index_t i0, index_t i1, index_t i2, index_t i3)  { Resize(type, frame_size, indices_t({i0, i1, i2, i3})); }
    
   	void Reshape(std::vector<index_t> shape)
	{
        index_t auto_index = -1;
		index_t total = 1;
        for (index_t i = 0; i < (index_t)shape.size(); ++i)
        {
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
        tensor_shape.push_back(-1);
        for ( auto size : shape ) {
            tensor_shape.push_back(size);
        }

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

//	void FillZeroMargin(void)
//	{
//	}
    

	int     GetType(void)  const { return m_data_type; }
	index_t GetFrameSize(void)  const { return m_frame_size; }
	index_t GetNodeSize(void)  const { return m_node_size; }


    // ---------------------------------
    //  ダイレクトアクセス用
    // ---------------------------------

	index_t GetFrameStride(void)  const { return m_frame_stride; }

    Memory::Ptr         GetMemoryPtr(bool new_buf=false) const    { return m_tensor.GetMemoryPtr(new_buf); }
    Memory::ConstPtr    GetMemoryConstPtr(void) const             { return m_tensor.GetMemoryConstPtr(); }
    Memory::DevPtr      GetMemoryDevPtr(bool new_buf=false) const { return m_tensor.GetMemoryDevPtr(new_buf); }
    Memory::DevConstPtr GetMemoryDevConstPtr(void) const          { return m_tensor.GetMemoryDevConstPtr(); }

    // 型指定アクセス
   	template <typename MemTp, typename ValueTp>
	inline ValueTp Get(void const *addr, index_t frame, index_t node) const
	{
        BB_DEBUG_ASSERT(m_data_type == DataType<MemTp>::type);
        auto ptr = GetMemoryConstPtr();
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
        auto ptr = GetMemoryPtr();
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
        auto ptr = GetMemoryPtr();
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
    // ---------------------------------
    //  アクセス用
    // ---------------------------------

    void *GetNodeBaseAddr(void* base_addr, index_t node) const
    {
        auto addr = (std::uint8_t*)base_addr;
        return addr + (m_frame_stride * node);
    }

    void const *GetNodeBaseAddr(const void* base_addr, index_t node) const
    {
        auto addr = (std::uint8_t const *)base_addr;
        return addr + (m_frame_stride * node);
    }

    inline index_t GetNodeIndex(std::vector<index_t> const & indices) const
    {
        BB_DEBUG_ASSERT(indices.size() == m_node_shape.size());

        index_t stride = 1;
        index_t index = 0;
        for ( index_t i = 0; i < (index_t)m_node_shape.size(); ++i ) {
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
		case BB_TYPE_FP32:   return static_cast<Tp>(DataType_Read<float>       (base, frame));	break;
		case BB_TYPE_FP64:   return static_cast<Tp>(DataType_Read<double>      (base, frame));	break;
        case BB_TYPE_INT8:   return static_cast<Tp>(DataType_Read<std::int8_t> (base, frame));  break;
		case BB_TYPE_INT16:  return static_cast<Tp>(DataType_Read<std::int16_t>(base, frame));	break;
   		case BB_TYPE_INT32:  return static_cast<Tp>(DataType_Read<std::int32_t>(base, frame));	break;
		case BB_TYPE_INT64:  return static_cast<Tp>(DataType_Read<std::int64_t>(base, frame));	break;
        case BB_TYPE_UINT8:  return static_cast<Tp>(DataType_Read<std::int8_t> (base, frame));  break;
		case BB_TYPE_UINT16: return static_cast<Tp>(DataType_Read<std::int16_t>(base, frame));	break;
   		case BB_TYPE_UINT32: return static_cast<Tp>(DataType_Read<std::int32_t>(base, frame));	break;
		case BB_TYPE_UINT64: return static_cast<Tp>(DataType_Read<std::int64_t>(base, frame));	break;
        default:   BB_ASSERT(0);
        }
		return 0;
	}

    template<typename Tp>
    void WriteValue(void *base, index_t frame, Tp value) const
	{
		switch (m_data_type) {
		case BB_TYPE_BIT:    DataType_Write<Bit>         (base, frame, static_cast<Bit>     (value));   break;
		case BB_TYPE_FP32:   DataType_Write<float>       (base, frame, static_cast<float>   (value));	break;
		case BB_TYPE_FP64:   DataType_Write<double>      (base, frame, static_cast<double>  (value));	break;
        case BB_TYPE_INT8:   DataType_Write<std::int8_t> (base, frame, static_cast<int8_t>  (value));   break;
		case BB_TYPE_INT16:  DataType_Write<std::int16_t>(base, frame, static_cast<int16_t> (value));	break;
   		case BB_TYPE_INT32:  DataType_Write<std::int32_t>(base, frame, static_cast<int32_t> (value));	break;
		case BB_TYPE_INT64:  DataType_Write<std::int64_t>(base, frame, static_cast<int64_t> (value));	break;
        case BB_TYPE_UINT8:  DataType_Write<std::int8_t> (base, frame, static_cast<uint8_t> (value));   break;
		case BB_TYPE_UINT16: DataType_Write<std::int16_t>(base, frame, static_cast<uint16_t>(value));	break;
   		case BB_TYPE_UINT32: DataType_Write<std::int32_t>(base, frame, static_cast<uint32_t>(value));	break;
		case BB_TYPE_UINT64: DataType_Write<std::int64_t>(base, frame, static_cast<uint64_t>(value));	break;
        default:   BB_ASSERT(0);
        }
	}

    template<typename Tp>
    void AddValue(void *base, index_t frame, Tp value) const
	{
		switch (m_data_type) {
		case BB_TYPE_BIT:    DataType_Add<Bit>         (base, frame, static_cast<Bit>     (value)); break;
		case BB_TYPE_FP32:   DataType_Add<float>       (base, frame, static_cast<float>   (value));	break;
		case BB_TYPE_FP64:   DataType_Add<double>      (base, frame, static_cast<double>  (value)); break;
        case BB_TYPE_INT8:   DataType_Add<std::int8_t> (base, frame, static_cast<int8_t>  (value)); break;
		case BB_TYPE_INT16:  DataType_Add<std::int16_t>(base, frame, static_cast<int16_t> (value));	break;
   		case BB_TYPE_INT32:  DataType_Add<std::int32_t>(base, frame, static_cast<int32_t> (value));	break;
		case BB_TYPE_INT64:  DataType_Add<std::int64_t>(base, frame, static_cast<int64_t> (value));	break;
        case BB_TYPE_UINT8:  DataType_Add<std::int8_t> (base, frame, static_cast<uint8_t> (value)); break;
		case BB_TYPE_UINT16: DataType_Add<std::int16_t>(base, frame, static_cast<uint16_t>(value));	break;
   		case BB_TYPE_UINT32: DataType_Add<std::int32_t>(base, frame, static_cast<uint32_t>(value));	break;
		case BB_TYPE_UINT64: DataType_Add<std::int64_t>(base, frame, static_cast<uint64_t>(value));	break;
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
    auto GetConstPtr(void) const
    {
        FrameBufferConstPtr_<Tp const, FrameBuffer const, Memory::ConstPtr> ptr(this);
        ptr.Lock();
        return ptr;
    }

    template <typename Tp>
    auto GetPtr(bool new_buf=false)
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
        auto ptr = GetMemoryConstPtr();
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
        auto ptr = GetMemoryPtr();
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
        auto ptr = GetMemoryPtr();
        DataType_Add<MemTp>(GetNodeBaseAddr(ptr.GetAddr(), node), frame, static_cast<MemTp>(value));
	}

   	template <typename MemTp, typename ValueTp>
	inline void Add(index_t frame, std::vector<index_t> const & indices, ValueTp value)
	{
        Add<MemTp, ValueTp>(frame, GetNodeIndex(indices), value);
	}


    // 汎用アクセス
    template <typename Tp>
	inline Tp GetValue(index_t frame, index_t node)
	{
        auto ptr = GetMemoryPtr();
        return ReadValue<Tp>(GetNodeBaseAddr(ptr.GetAddr(), node), frame);
	}

   	template <typename Tp>
    inline Tp GetValue(index_t frame, std::vector<index_t> const & indices)
    {
        return GetValue<Tp>(frame, GetNodeIndex(indices));
    }

   	template <typename Tp>
	inline void SetValue(index_t frame, index_t node, Tp value)
	{
        auto ptr = GetMemoryPtr();
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
        auto ptr = GetMemoryPtr();
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

    inline Bit           GetBit   (index_t frame, index_t node) { return GetValue<Bit          >(frame, node); }
    inline float         GetFP32  (index_t frame, index_t node) { return GetValue<float        >(frame, node); }
    inline double        GetFP64  (index_t frame, index_t node) { return GetValue<double       >(frame, node); }
    inline std::int8_t   GetINT8  (index_t frame, index_t node) { return GetValue<std::int8_t  >(frame, node); }
    inline std::int16_t  GetINT16 (index_t frame, index_t node) { return GetValue<std::int16_t >(frame, node); }
    inline std::int32_t  GetINT32 (index_t frame, index_t node) { return GetValue<std::int32_t >(frame, node); }
    inline std::int64_t  GetINT64 (index_t frame, index_t node) { return GetValue<std::int64_t >(frame, node); }
    inline std::uint8_t  GetUINT8 (index_t frame, index_t node) { return GetValue<std::uint8_t >(frame, node); }
    inline std::uint16_t GetUINT16(index_t frame, index_t node) { return GetValue<std::uint16_t>(frame, node); }
    inline std::uint32_t GetUINT32(index_t frame, index_t node) { return GetValue<std::uint32_t>(frame, node); }
    inline std::uint64_t GetUINT64(index_t frame, index_t node) { return GetValue<std::uint64_t>(frame, node); }
    
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
        BB_ASSERT(frame > 0 && frame < m_frame_size);

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
        BB_ASSERT(offset + m_frame_size < (index_t)data.size() );
        for (index_t frame = 0; frame < m_frame_size; ++frame) {
            BB_ASSERT(data[frame].size() == (size_t)m_node_size);
            for (index_t node = 0; node < m_node_size; ++node) {
                SetValue<Tp>(frame, node, data[frame + offset][node]);
            }
        }
    }

    // テンソルの設定
public:
    template<typename Tp>
    void SetTensor(index_t frame, Tensor const &tensor)
    {
        BB_ASSERT(m_data_type == DataType<Tp>::type);
        BB_ASSERT(tensor.GetType() == DataType<Tp>::type);
        BB_ASSERT(tensor.GetShape() == m_tensor.GetShape());
        
        int dim = tensor.GetDim();
        std::vector<index_t> indices(tensor.GetDim(), 0);
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


    // 部分切り出し
    FrameBuffer GetRange(index_t start, index_t size)
    {
        BB_ASSERT(start >= 0 && start < m_frame_size);
        BB_ASSERT(size >= 0 &&  size < m_frame_size - start);

        FrameBuffer buf(m_data_type, size, m_node_shape);

        auto src_ptr = m_tensor.GetMemoryConstPtr();
        auto dst_ptr = buf.m_tensor.GetMemoryPtr(true);
        auto src_addr = (std::int8_t const *)src_ptr.GetAddr();
        auto dst_addr = (std::int8_t       *)dst_ptr.GetAddr();

        if (m_data_type == BB_TYPE_BIT && (start % 8) != 0 ) {
            #pragma omp parallel for
            for (index_t node = 0; node < m_node_size; ++node)
            {
                for (index_t frame = 0; frame < size; ++frame) {
                    auto val = DataType_Read<Bit>(src_addr + m_frame_stride * node, frame + start);
                    DataType_Write<Bit>(dst_addr + m_frame_stride * node, frame, val);
                }
            }           
        }
        else {
            int     unit   = DataType_GetBitSize(m_data_type);
            index_t byte_offset = (start * unit + 7) / 8;
            index_t byte_size   = (size * unit + 7) / 8;

            #pragma omp parallel for
            for (index_t node = 0; node < m_node_size; ++node)
            {
                memcpy(dst_addr + buf.m_frame_stride * node, src_addr + m_frame_stride * node + byte_offset, byte_size);
            }
        }

        return buf;
    }

};



}
