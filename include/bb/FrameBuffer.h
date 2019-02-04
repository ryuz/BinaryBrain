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
template <typename T = float>
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
		FrameBuffer clone_buf();

        clone_buf.m_tensor  = m_tensor.Clone();
		m_data_type         = buf.m_data_type;
		m_node_size = buf.m_node_size;
		m_frame_size = buf.m_frame_size;
		m_frame_stride = buf.m_frame_stride;
        m_shape        = m_shape;

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

        // tensor の shape設定
        int                     tensor_type = data_type;
        std::vector<index_t>    tensor_shape(shape.size() + 1);

        // Bit型は内部 UINT8 で扱う
        if ( data_type == BB_TYPE_BIT )
        {
            tensor_type = BB_TYPE_UINT8;
        }

        // サイズ計算
		m_node_size = 1;
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
	std::vector<index_t> GetShape(void) const
	{
		return m_node_shape;
	}


	index_t GetFrameStride(void)  const { return m_frame_stride; }
    
    Memory::Ptr GetPtr(bool new_buf=false) const { return m_tensor.GetPtr(new_buf); }
    Memory::Ptr GetConstPtr(void) const { return m_tensor.GetConstPtr(); }
    Memory::Ptr GetDevPtr(bool new_buf=false) const { return m_tensor.GetDevPtr(new_buf); }
    Memory::Ptr GetDevConstPtr(void) const { return m_tensor.GetDevConstPtr(); }


protected:
	inline void* GetBasePtr(INDEX addr) const
	{
		return &m_buffer.get()[m_frame_stride * addr];
	}

	/*
	template <typename Tp>
	inline void Write(void *base, INDEX frame, Tp value) const
	{
		Tp* ptr = (Tp*)base;
		ptr[frame] = value;
	}

	template <>
	inline void Write(void *base, INDEX frame, bool value) const
	{
		std::uint8_t* ptr = (std::uint8_t*)base;
		std::uint8_t mask = (std::uint8_t)(1 << (frame % 8));
		if (value) {
			ptr[frame / 8] |= mask;
		}
		else {
			ptr[frame / 8] &= ~mask;
		}
	}
		
	template <>
	inline void Write(void *base, INDEX frame, Bit value) const
	{
		Write<bool>(base, frame, (bool)value);
	}

	template <typename Tp>
	inline Tp Read(void *base, INDEX frame) const
	{
		Tp* ptr = (Tp*)base;
		return ptr[frame];
	}

	template <>
	inline bool Read<bool>(void *base, INDEX frame) const
	{
		std::uint8_t* ptr = (std::uint8_t*)base;
		std::uint8_t mask = (std::uint8_t)(1 << (frame % 8));
		return ((ptr[frame / 8] & mask) != 0);
	}

	template <>
	inline Bit Read<Bit>(void *base, INDEX frame) const
	{
		return Read<bool>(base, frame);
	}
	*/


	inline void WriteReal(void *base, INDEX frame, T value) const
	{
		switch (m_data_type) {
		case BB_TYPE_BINARY: NeuralNet_Write<bool>(base, frame, value > (T)0.5);	break;
		case BB_TYPE_REAL32: NeuralNet_Write<float>(base, frame, (float)value);	break;
		case BB_TYPE_REAL64: NeuralNet_Write<double>(base, frame, (double)value);	break;
		}
	}

	inline T ReadReal(void *base, INDEX frame) const
	{
		switch (m_data_type) {
		case BB_TYPE_BINARY: return NeuralNet_Read<bool>(base, frame) ? (T)1.0 : (T)0.0;
		case BB_TYPE_REAL32: return (T)NeuralNet_Read<float>(base, frame);
		case BB_TYPE_REAL64: return (T)NeuralNet_Read<double>(base, frame);
		}
		return 0;
	}

	inline void WriteBinary(void *base, INDEX frame, bool value) const
	{
		switch (m_data_type) {
		case BB_TYPE_BINARY: NeuralNet_Write<bool>(base, frame, value);	break;
		case BB_TYPE_REAL32: NeuralNet_Write<float>(base, frame, (value ? 1.0f : 0.0f)); 	break;
		case BB_TYPE_REAL64: NeuralNet_Write<double>(base, frame, (value ? 1.0 : 0.0)); 	break;
		}
	}

	inline bool ReadBinary(void *base, INDEX frame) const
	{
		switch (m_data_type) {
		case BB_TYPE_BINARY: return NeuralNet_Read<bool>(base, frame);
		case BB_TYPE_REAL32: return (NeuralNet_Read<float>(base, frame) > 0.5f);
		case BB_TYPE_REAL64: return (NeuralNet_Read<double>(base, frame) > 0.5);
		}
		return false;
	}


public:
	inline void* GetZeroPtr(void) const
	{
		return GetBasePtr(m_base_size);
	}

#if	BB_NEURALNET_BUFFER_USE_ROI
	inline void* GetPtr(INDEX node) const
	{
		INDEX addr = 0;
		for (size_t i = 0; i < m_dim.size(); ++i) {
			INDEX index = node % m_dim[i].width;
			addr += m_dim[i].stride * (index + m_dim[i].offset);
			node /= m_dim[i].width;
		}
		return GetBasePtr(addr);
	}

	inline void* GetPtr(std::vector<INDEX> index) const
	{
		BB_ASSERT(index.size() == m_dim.size());
		INDEX addr = 0;
		for (size_t i = 0; i < m_dim.size(); ++i) {
			BB_ASSERT(index[i] >= 0 && index[i] < m_dim[i].width);
			addr += m_dim[i].stride * (index[i] + m_dim[i].offset);
		}

		return GetBasePtr(addr);
	}

	template <size_t N>
	inline void* GetPtrN(std::array<INDEX, N> index) const
	{
		BB_ASSERT(index.size() == m_dim.size());
		INDEX addr = 0;
		for (size_t i = 0; i < index.size(); ++i) {
			addr += m_dim[i].stride * (index[i] + m_dim[i].offset);
		}

		return GetBasePtr(addr);
	}
#else
	inline void* GetPtr(INDEX node) const
	{
		return GetBasePtr(node);
	}

	inline void* GetPtr(std::vector<INDEX> index) const
	{
		BB_ASSERT(index.size() == m_dim.size());
		INDEX addr = 0;
		for (size_t i = 0; i < m_dim.size(); ++i) {
			BB_ASSERT(index[i] >= 0 && index[i] < m_dim[i].step);
			addr += m_dim[i].stride * index[i];
		}

		return GetBasePtr(addr);
	}

	template <size_t N>
	inline void* GetPtrN(std::array<INDEX, N> index) const
	{
		BB_ASSERT(index.size() == m_dim.size());
		INDEX addr = 0;
		for (size_t i = 0; i < index.size(); ++i) {
			addr += m_dim[i].stride * index[i];
		}

		return GetBasePtr(addr);
	}
#endif

	inline void* GetPtr2(INDEX i1, INDEX i0) const
	{
		return GetPtrN<2>({ i0 , i1 });
	}

	inline void* GetPtr3(INDEX i2, INDEX i1, INDEX i0) const
	{
		return GetPtrN<3>({ i0, i1, i2 });
	}

	inline void* GetPtr4(INDEX i3, INDEX i2, INDEX i1, INDEX i0) const
	{
		return GetPtrN<4>({ i0, i1, i2, i3 });
	}

	inline void* GetPtr5(INDEX i4, INDEX i3, INDEX i2, INDEX i1, INDEX i0) const
	{
		return GetPtrN<5>({ i0, i1, i2, i3, i4 });
	}


	inline void ResetPtr(void)
	{
		m_end = false;
		m_iterator.resize(m_dim.size());
		std::fill(m_iterator.begin(), m_iterator.end(), 0);
	}

#if BB_NEURALNET_BUFFER_USE_ROI
	inline void* NextPtr(void)
	{
		void* ptr = GetPtr(m_iterator);

		for (int i = 0; i < m_iterator.size(); ++i) {
			++m_iterator[i];
			if (m_iterator[i] < m_dim[i].width) {
				return ptr;
			}
			m_iterator[i] = 0;
		}
		m_end = true;
		return ptr;
	}
#else
	inline void* NextPtr(void)
	{
		void* ptr = GetPtr(m_iterator);

		for (int i = 0; i < m_iterator.size(); ++i) {
			++m_iterator[i];
			if (m_iterator[i] < m_dim[i].step) {
				return ptr;
			}
			m_iterator[i] = 0;
		}
		m_end = true;
		return ptr;
	}
#endif

	inline bool IsEnd(void) const
	{
		return m_end;
	}


	template <typename Tp>
	inline void Set(INDEX frame, INDEX node, Tp value) const
	{
		NeuralNet_Write<Tp>(GetPtr(node), frame, value);
	}

	template <typename Tp>
	inline Tp Get(INDEX frame, INDEX node) const
	{
		return NeuralNet_Read<Tp>(GetPtr(node), frame);
	}


	void SetReal(INDEX frame, INDEX node, T value) const {	WriteReal(GetPtr(node), frame, value); }
	void SetReal(INDEX frame, std::vector<INDEX> index, T value) const { WriteReal(GetPtr(index), frame, value); }

	T GetReal(INDEX frame, INDEX node) const { return ReadReal(GetPtr(node), frame); }
	T GetReal(INDEX frame, std::vector<INDEX> index) const { return ReadReal(GetPtr(index), frame); }

	void SetBinary(INDEX frame, INDEX node, bool value) const { WriteBinary(GetPtr(node), frame, value); }
	void SetBinary(INDEX frame, std::vector<INDEX> index, bool value) const { WriteBinary(GetPtr(index), frame, value); }

	bool GetBinary(INDEX frame, INDEX node) const { return ReadBinary(GetPtr(node), frame); }
	bool GetBinary(INDEX frame, std::vector<INDEX> index) const { return ReadBinary(GetPtr(index), frame); }


//	friend std::ostream& operator<<(std::ostream& os, const NeuralNetBuffer<T>& buf);
};




/*
template <typename T = float>
std::ostream& operator<<(std::ostream& os, const NeuralNetBuffer<T>& buf)
{
	auto out_stream = [&out_stream](std::ostream& os, const NeuralNetBuffer<T>& buf, std::vector<INDEX>& idx, INDEX depth)
	{
		if (depth == 0) {
			os << "[";
			for (INDEX i = 0; i < buf.m_dim[depth]; ++i) {
				idx[depth] = i;
				os << buf.GetReal(idx) << ", ";
			}
			os << "]" << std::endl;
			return;
		}
		else {
			os << "[";
			for (INDEX i = 0; i < buf.m_dim[depth]; ++i) {
				idx[depth] = i;
				out_stream(os, buf, idx, depth - 1);
			}
			os << "]";
		}
	};

	std::vector<INDEX> idx(buf, m_dim.size(), 0);
	out_stream(os, buf, idx, (INDEX)(m_dim.size() - 1));

	return os;
}
*/


}
