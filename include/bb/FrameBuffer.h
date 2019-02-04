// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once

#include <stdlib.h>
#include <malloc.h>

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
    // デフォルトコンストラクタ
	FrameBuffer() {}

    // コピーコンストラクタ
	FrameBuffer(const FrameBuffer& buf)
	{
		*this = buf;
	}

  	/**
     * @brief  コンストラクタ
     * @detail コンストラクタ
     *        tensor は node_size サイズの1次元で初期化
     * @param frame_size フレーム数
     * @param node_size  1フレームのノード数
	 * @param data_type  1ノードのデータ型
     */
	FrameBuffer(index_t frame_size, index_t node_size, int data_type)
	{
        Resize(frame_size, {node_size}, data_type);
	}

   	/**
     * @brief  コンストラクタ
     * @detail コンストラクタ
     * @param frame_size フレーム数
     * @param shape      1フレームのノードを構成するshape
	 * @param data_type  1ノードのデータ型
     */
	FrameBuffer(index_t frame_size, std::vector<index_t> shape, int data_type)
	{
		Resize(frame_size, shape, data_type);
	}
    
   	/**
     * @brief  代入演算子
     * @detail 代入演算子
     *         代入演算子でのコピーは、メモリは同じ箇所を指す
     */
    FrameBuffer& operator=(const FrameBuffer &buf)
	{
		m_tensor        = buf.m_tensor;
		m_data_type     = buf.m_data_type;
		m_frame_size    = buf.m_frame_size;
		m_frame_stride  = buf.m_frame_stride;
		m_node_size     = buf.m_node_size;
        m_node_shape    = m_node_shape;

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
     * @brief  サイズ設定
     * @detail サイズ設定
     * @param frame_size フレーム数
     * @param shape      1フレームのノードを構成するshape
	 * @param data_type  1ノードのデータ型
     */
    void Resize(index_t frame_size, std::vector<index_t> shape, int data_type)
	{
        m_data_type    = data_type;
        m_frame_size   = frame_size;
        m_frame_stride = ((DataType_GetBitSize(data_type) + 255) / 256) * (255 / 8);        // frame軸は256bit境界にあわせる(SIMD命令用)
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
		m_tensor.Resize(tensor_shape, tensor_type);
	}

   	/**
     * @brief  サイズ設定
     * @detail サイズ設定
     * @param frame_size フレーム数
     * @param node_size  1フレームのノードサイズ
	 * @param data_type  1ノードのデータ型
     */	void Resize(index_t frame_size, index_t node_size, int data_type)
	{
        std::vector<index_t>    shape(1);
        shape[0] = node_size;
        Resize(frame_size, shape, data_type);
	}

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

    Memory::Ptr         GetPtr(bool new_buf=false) const { return m_tensor.GetPtr(new_buf); }
    Memory::ConstPtr    GetConstPtr(void) const { return m_tensor.GetConstPtr(); }
    Memory::DevPtr      GetDevPtr(bool new_buf=false) const { return m_tensor.GetDevPtr(new_buf); }
    Memory::DevConstPtr GetDevConstPtr(void) const { return m_tensor.GetDevConstPtr(); }



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

    inline index_t GetNodeIndex(std::vector<index_t> const & indexces) const
    {
        BB_DEBUG_ASSERT(indexces.size() == m_node_shape.size());

        index_t stride = 1;
        index_t index = 0;
        for ( index_t i = 0; i < (index_t)m_node_shape.size(); ++i ) {
            BB_DEBUG_ASSERT(indexces[i] >= 0 && indexces[i] < m_node_shape[i]);
            index += stride * indexces[i];
            stride *= m_node_shape[i];
        }
        return index;
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


public:

    // 高速アクセス
	template <typename MemTp, typename ValueTp>
	inline void Set(index_t frame, index_t node, ValueTp value)
	{
        BB_DEBUG_ASSERT(m_data_type == DataType<MemTp>::type);
        auto ptr = GetPtr();
        DataType_Write<MemTp>(GetNodeBaseAddr(ptr.GetAddr(), node), frame, static_cast<MemTp>(value));
	}

   	template <typename MemTp, typename ValueTp>
	inline void Set(index_t frame, std::vector<index_t> const & indexces, ValueTp value)
	{
        Set<MemTp, ValueTp>(frame, GetNodeIndex(indexces), value);
	}

	template <typename MemTp, typename ValueTp>
	inline ValueTp Get(index_t frame, index_t node) const
	{
        BB_DEBUG_ASSERT(m_data_type == DataType<MemTp>::type);
        auto ptr = GetConstPtr();
		return static_cast<ValueTp>(DataType_Read<MemTp>(GetNodeBaseAddr(ptr.GetAddr(), node), frame)); 
	}
    
   	template <typename MemTp, typename ValueTp>
	inline ValueTp Get(index_t frame, std::vector<index_t> const & indexces)
	{
        return Get<MemTp, ValueTp>(frame, GetNodeIndex(indexces));
	}


    // 汎用アクセス
  	template <typename Tp>
	inline void SetValue(index_t frame, index_t node, Tp value)
	{
        auto ptr = GetPtr();
        WriteValue<Tp>(GetNodeBaseAddr(ptr.GetAddr(), node), frame, value);
	}

   	template <typename Tp>
    inline void SetValue(index_t frame, std::vector<index_t> const & indexces, Tp value)
    {
        SetValue<Tp>(frame, GetNodeIndex(indexces), value);
    }

    template <typename Tp>
	inline Tp GetValue(index_t frame, index_t node)
	{
        auto ptr = GetPtr();
        return ReadValue<Tp>(GetNodeBaseAddr(ptr.GetAddr(), node), frame);
	}

   	template <typename Tp>
    inline Tp GetValue(index_t frame, std::vector<index_t> const & indexces)
    {
        return GetValue<Tp>(frame, GetNodeIndex(indexces));
    }

    inline void SetValueBit   (index_t frame, index_t node, Bit           value) { SetValue<Bit          >(frame, node, value); }
    inline void SetValueFP32  (index_t frame, index_t node, float         value) { SetValue<float        >(frame, node, value); }
    inline void SetValueFP64  (index_t frame, index_t node, double        value) { SetValue<double       >(frame, node, value); }
    inline void SetValueINT8  (index_t frame, index_t node, std::int8_t   value) { SetValue<std::int8_t  >(frame, node, value); }
    inline void SetValueINT16 (index_t frame, index_t node, std::int16_t  value) { SetValue<std::int16_t >(frame, node, value); }
    inline void SetValueINT32 (index_t frame, index_t node, std::int32_t  value) { SetValue<std::int32_t >(frame, node, value); }
    inline void SetValueINT64 (index_t frame, index_t node, std::int64_t  value) { SetValue<std::int64_t >(frame, node, value); }
    inline void SetValueUINT8 (index_t frame, index_t node, std::uint8_t  value) { SetValue<std::uint8_t >(frame, node, value); }
    inline void SetValueUINT16(index_t frame, index_t node, std::uint16_t value) { SetValue<std::uint16_t>(frame, node, value); }
    inline void SetValueUINT32(index_t frame, index_t node, std::uint32_t value) { SetValue<std::uint32_t>(frame, node, value); }
    inline void SetValueUINT64(index_t frame, index_t node, std::uint64_t value) { SetValue<std::uint64_t>(frame, node, value); }

    inline void SetValueBit   (index_t frame, std::vector<index_t> const & indexces, Bit           value) { SetValue<Bit          >(frame, indexces, value); }
    inline void SetValueFP32  (index_t frame, std::vector<index_t> const & indexces, float         value) { SetValue<float        >(frame, indexces, value); }
    inline void SetValueFP64  (index_t frame, std::vector<index_t> const & indexces, double        value) { SetValue<double       >(frame, indexces, value); }
    inline void SetValueINT8  (index_t frame, std::vector<index_t> const & indexces, std::int8_t   value) { SetValue<std::int8_t  >(frame, indexces, value); }
    inline void SetValueINT16 (index_t frame, std::vector<index_t> const & indexces, std::int16_t  value) { SetValue<std::int16_t >(frame, indexces, value); }
    inline void SetValueINT32 (index_t frame, std::vector<index_t> const & indexces, std::int32_t  value) { SetValue<std::int32_t >(frame, indexces, value); }
    inline void SetValueINT64 (index_t frame, std::vector<index_t> const & indexces, std::int64_t  value) { SetValue<std::int64_t >(frame, indexces, value); }
    inline void SetValueUINT8 (index_t frame, std::vector<index_t> const & indexces, std::uint8_t  value) { SetValue<std::uint8_t >(frame, indexces, value); }
    inline void SetValueUINT16(index_t frame, std::vector<index_t> const & indexces, std::uint16_t value) { SetValue<std::uint16_t>(frame, indexces, value); }
    inline void SetValueUINT32(index_t frame, std::vector<index_t> const & indexces, std::uint32_t value) { SetValue<std::uint32_t>(frame, indexces, value); }
    inline void SetValueUINT64(index_t frame, std::vector<index_t> const & indexces, std::uint64_t value) { SetValue<std::uint64_t>(frame, indexces, value); }

    inline Bit           GetValueBit   (index_t frame, index_t node) { return GetValue<Bit          >(frame, node); }
    inline float         GetValueFP32  (index_t frame, index_t node) { return GetValue<float        >(frame, node); }
    inline double        GetValueFP64  (index_t frame, index_t node) { return GetValue<double       >(frame, node); }
    inline std::int8_t   GetValueINT8  (index_t frame, index_t node) { return GetValue<std::int8_t  >(frame, node); }
    inline std::int16_t  GetValueINT16 (index_t frame, index_t node) { return GetValue<std::int16_t >(frame, node); }
    inline std::int32_t  GetValueINT32 (index_t frame, index_t node) { return GetValue<std::int32_t >(frame, node); }
    inline std::int64_t  GetValueINT64 (index_t frame, index_t node) { return GetValue<std::int64_t >(frame, node); }
    inline std::uint8_t  GetValueUINT8 (index_t frame, index_t node) { return GetValue<std::uint8_t >(frame, node); }
    inline std::uint16_t GetValueUINT16(index_t frame, index_t node) { return GetValue<std::uint16_t>(frame, node); }
    inline std::uint32_t GetValueUINT32(index_t frame, index_t node) { return GetValue<std::uint32_t>(frame, node); }
    inline std::uint64_t GetValueUINT64(index_t frame, index_t node) { return GetValue<std::uint64_t>(frame, node); }
    
    inline Bit           GetValueBit   (index_t frame, std::vector<index_t> const & indexces) { return GetValue<Bit          >(frame, indexces); }
    inline float         GetValueFP32  (index_t frame, std::vector<index_t> const & indexces) { return GetValue<float        >(frame, indexces); }
    inline double        GetValueFP64  (index_t frame, std::vector<index_t> const & indexces) { return GetValue<double       >(frame, indexces); }
    inline std::int8_t   GetValueINT8  (index_t frame, std::vector<index_t> const & indexces) { return GetValue<std::int8_t  >(frame, indexces); }
    inline std::int16_t  GetValueINT16 (index_t frame, std::vector<index_t> const & indexces) { return GetValue<std::int16_t >(frame, indexces); }
    inline std::int32_t  GetValueINT32 (index_t frame, std::vector<index_t> const & indexces) { return GetValue<std::int32_t >(frame, indexces); }
    inline std::int64_t  GetValueINT64 (index_t frame, std::vector<index_t> const & indexces) { return GetValue<std::int64_t >(frame, indexces); }
    inline std::uint8_t  GetValueUINT8 (index_t frame, std::vector<index_t> const & indexces) { return GetValue<std::uint8_t >(frame, indexces); }
    inline std::uint16_t GetValueUINT16(index_t frame, std::vector<index_t> const & indexces) { return GetValue<std::uint16_t>(frame, indexces); }
    inline std::uint32_t GetValueUINT32(index_t frame, std::vector<index_t> const & indexces) { return GetValue<std::uint32_t>(frame, indexces); }
    inline std::uint64_t GetValueUINT64(index_t frame, std::vector<index_t> const & indexces) { return GetValue<std::uint64_t>(frame, indexces); }


    // テンソルの設定
protected:
    template<typename Tp>
    void SetTensor(index_t frame, Tensor const &tensor)
    {
        BB_ASSERT(m_data_type == DataType<Tp>::type);
        BB_ASSERT(tensor.GetType() == DataType<Tp>::type);


    }
};


}
