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

#include <intrin.h>
#include <mmintrin.h>
#include <immintrin.h>

#include "NeuralNetType.h"
#include "NeuralNetUtility.h"


namespace bb {


// 色々な型のデータを管理することを目的としたバッファ
// イメージとしては OpenCV の Mat型 のような汎用性を目指す
//
// メモリはベースを2次元として、画像フレーム毎にバッチ処理する場合の
// frame 軸と、各層の演算ノードに対応する node 軸を持っている
// frame軸は、SIMD演算を意識して32バイト境界を守り、バイナリ値は
// __m256i に 256bit パッキングする
// node 軸はさらに必要に応じて、テンソル的に多次元化可能にしておき、
// 転置や reshape、畳み込み時のROIアクセスなど考慮に入れておく


#define BB_NEURALNET_BUFFER_USE_ROI		0


// NeuralNet用のバッファ
template <typename T = float, typename INDEX = size_t>
class NeuralNetBuffer
{
protected:
	std::shared_ptr<std::uint8_t>	m_buffer;
	int								m_data_type = 0;
	INDEX							m_base_size = 0;
	INDEX							m_frame_size = 0;
	INDEX							m_frame_stride = 0;

	struct Dimension
	{
		INDEX	step;
		INDEX	stride;

#if	BB_NEURALNET_BUFFER_USE_ROI
		INDEX	offset;
		INDEX	width;
#endif
	};

	std::vector<Dimension>			m_dim;

	INDEX							m_node_size = 0;
	std::vector<INDEX>				m_iterator;
	bool							m_end;

public:
	NeuralNetBuffer() {}
	NeuralNetBuffer(const NeuralNetBuffer& buf)
	{
		*this = buf;
	}

	NeuralNetBuffer(INDEX frame_size, INDEX node_size, int data_type)
	{
		Resize(frame_size, node_size, data_type);
	}

	NeuralNetBuffer& operator=(const NeuralNetBuffer &buf)
	{
		m_buffer = buf.m_buffer;
		m_data_type = buf.m_data_type;
		m_base_size = buf.m_base_size;
		m_frame_size = buf.m_frame_size;
		m_frame_stride = buf.m_frame_stride;

		m_dim = buf.m_dim;
		m_node_size = buf.m_node_size;
		m_iterator = buf.m_iterator;
		m_end = buf.m_end;

		return *this;
	}

	NeuralNetBuffer clone(void) const
	{
		NeuralNetBuffer clone_buf(m_frame_size, m_base_size, m_data_type);

		memcpy(clone_buf.m_buffer.get(), m_buffer.get(), m_frame_stride*(m_base_size+1));

		clone_buf.m_frame_size = m_frame_size;
		clone_buf.m_frame_stride = m_frame_stride;

		clone_buf.m_dim = m_dim;
		clone_buf.m_node_size = m_node_size;
		clone_buf.m_iterator = m_iterator;
		clone_buf.m_end = m_end;

		return clone_buf;
	}

	void Clear(void)
	{
		memset(m_buffer.get(), 0, m_frame_stride*m_base_size);
	}


	void Resize(INDEX frame_size, INDEX node_size, int data_type)
	{
		// 設定保存
		m_data_type = data_type;
		m_base_size = node_size;
		m_frame_size = frame_size;

		size_t type_bit_size = NeuralNet_GetTypeBitSize(data_type);

		// メモリ確保
		m_frame_stride = (((frame_size * type_bit_size) + 255) / 256) * 32;
//#ifdef _MSC_VER
//		m_buffer = std::shared_ptr<std::uint8_t>((std::uint8_t *)_aligned_malloc(m_frame_stride*(m_base_size + 1), 32), _aligned_free);
//#else
//		m_buffer = std::shared_ptr<std::uint8_t>((std::uint8_t *)posix_memalign(m_frame_stride*(m_base_size + 1), 32), std::free);
//#endif
		m_buffer = std::shared_ptr<std::uint8_t>((std::uint8_t *)aligned_memory_alloc(m_frame_stride*(m_base_size + 1), 32), aligned_memory_free);

		memset(m_buffer.get(), 0, m_frame_stride*(m_base_size + 1));

		m_node_size = node_size;
		m_dim.resize(1);
		m_dim[0].step = node_size;
		m_dim[0].stride = 1;
#if	BB_NEURALNET_BUFFER_USE_ROI
		m_dim[0].offset = 0;
		m_dim[0].width = node_size;
#endif
	}

	void SetDimensions(std::vector<INDEX> dim)
	{
		BB_ASSERT(dim.size() > 0);
		INDEX total = 1; for (auto len : dim) { total *= len; }
		BB_ASSERT(total == m_base_size);

		m_dim.resize(dim.size());
		m_dim[0].step = dim[0];
		m_dim[0].stride = 1;
#if	BB_NEURALNET_BUFFER_USE_ROI
		m_dim[0].offset = 0;
		m_dim[0].width = dim[0];
#endif
		for (size_t i = 1; i < dim.size(); ++i) {
			m_dim[i].step = dim[i];
			m_dim[i].stride = m_dim[i - 1].stride * m_dim[i - 1].step;
#if	BB_NEURALNET_BUFFER_USE_ROI
			m_dim[i].offset = 0;
			m_dim[i].width = dim[i];
#endif
		}
	}

	std::vector<INDEX> GetDimensions(void) const
	{
		return m_dim;
	}

#if	BB_NEURALNET_BUFFER_USE_ROI
	void SetRoi(std::vector<INDEX> offset)
	{
		BB_ASSERT(offset.size() == m_dim.size());

		m_node_size = 1;
		for (size_t i = 0; i < m_dim.size(); ++i) {
			BB_ASSERT(m_dim[i].width > offset[i]);
			m_dim[i].offset += offset[i];
			m_dim[i].width -= offset[i];
			m_node_size *= m_dim[i].width;
		}
	}

	void SetRoi(std::vector<INDEX> offset, std::vector<INDEX> width)
	{
		BB_ASSERT(offset.size() == m_dim.size());
		BB_ASSERT(width.size() == m_dim.size());

		m_node_size = 1;
		for (size_t i = 0; i < m_dim.size(); ++i) {
			BB_ASSERT(m_dim[i].width > offset[i]);
			m_dim[i].offset += offset[i];
			m_dim[i].width = width[i];
			m_node_size *= m_dim[i].width;
			BB_ASSERT(m_dim[i].offset + m_dim[i].width <= m_dim[i].step);
		}
	}

	void ClearRoi(void)
	{
		for (auto& d : m_dim) {
			d.offset = 0;
			d.width = d.step;
		}
		m_node_size = m_base_size;
	}
#endif


	INDEX GetFrameSize(void)  const { return m_frame_size; }
	INDEX GetNodeSize(void)  const { return m_node_size; }
//	INDEX GetTypeBitSize(void)  const { return m_type_bit_size; }

	INDEX GetFrameStride(void)  const { return m_frame_stride; }

	void* GetBuffer(void) const
	{
		return m_buffer.get();
	}


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


//	friend std::ostream& operator<<(std::ostream& os, const NeuralNetBuffer<T, INDEX>& buf);
};




/*
template <typename T = float, typename INDEX = size_t>
std::ostream& operator<<(std::ostream& os, const NeuralNetBuffer<T, INDEX>& buf)
{
	auto out_stream = [&out_stream](std::ostream& os, const NeuralNetBuffer<T, INDEX>& buf, std::vector<INDEX>& idx, INDEX depth)
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
