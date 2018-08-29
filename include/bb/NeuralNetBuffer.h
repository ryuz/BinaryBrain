// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once

#include <memory>
#include <malloc.h>
#include <intrin.h>
#include "NeuralNetType.h"


namespace bb {


// NeuralNet用のバッファアクセサ
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
		INDEX	offset;
		INDEX	width;
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
		
		memcpy(clone_buf.m_buffer.get(), m_buffer.get(), m_frame_stride*m_base_size);

		clone_buf.m_frame_size = m_frame_size;
		clone_buf.m_frame_stride = m_frame_stride;

		clone_buf.m_dim = m_dim;
		clone_buf.m_node_size = m_node_size;
		clone_buf.m_iterator = m_iterator;
		clone_buf.m_end = m_end;

		return clone_buf;
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
		m_buffer = std::shared_ptr<std::uint8_t>((std::uint8_t *)_aligned_malloc(m_frame_stride*m_base_size, 32), _aligned_free);
		memset(m_buffer.get(), 0, m_frame_stride*m_base_size);

		m_node_size = node_size;
		m_dim.resize(1);
		m_dim[0].step   = node_size;
		m_dim[0].stride = 1;
		m_dim[0].offset = 0;
		m_dim[0].width  = node_size;
	}
	
	void SetDimensions(std::vector<INDEX> dim)
	{
		BB_ASSERT(dim.size() > 0);
		INDEX total = 1; for (auto len : dim) { total *= len; }
		BB_ASSERT(total == m_base_size);
		
		m_dim.resize(dim.size());
		m_dim[0].step = dim[0];
		m_dim[0].stride = 1;
		m_dim[0].offset = 0;
		m_dim[0].width = dim[0];
		for (size_t i = 1; i < dim.size(); ++i) {
			m_dim[i].step = dim[i];
			m_dim[i].stride = m_dim[i-1].stride * m_dim[i-1].step;
			m_dim[i].offset = 0;
			m_dim[i].width = dim[i];
		}
	}

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


	INDEX GetFrameSize(void)  const { return m_frame_size; }
	INDEX GetNodeSize(void)  const { return m_node_size; }
	INDEX GetTypeBitSize(void)  const { return m_type_bit_size; }
	
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

public:
	inline void* GetPtr(std::vector<INDEX> index) const
	{
		BB_ASSERT(index.size() == m_dim.size());
		INDEX addr = 0;
		for (size_t i = 0; i < m_dim.size(); ++i) {
			addr += m_dim[i].stride * (index[i] + m_dim[i].offset);
		}

		return GetBasePtr(addr);
	}

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

	inline void* GetPtr5(INDEX i4, INDEX i3, INDEX i2, INDEX i1, INDEX i0)
	{
		return GetPtrN<5>({ i0, i1, i2, i3, i4 }); const
	}


	inline void ResetPtr(void)
	{
		m_end = false;
		m_iterator.resize(m_dim.size());
		std::fill(m_iterator.begin(), m_iterator.end(), 0);
	}

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
	
	inline bool IsEnd(void) const
	{
		return m_end;
	}


	template <typename Tp>
	inline void Set(INDEX frame, INDEX node, Tp value) const
	{
		if (typeid(Tp) == typeid(bool)) {
//			std::uint8_t* ptr = &(m_buffer.get()[m_frame_stride * node]);
			std::uint8_t* ptr = (std::uint8_t*)GetPtr(node);
			std::uint8_t mask = (std::uint8_t)(1 << (frame % 8));
			if (value) {
				ptr[frame / 8] |= mask;
			}
			else {
				ptr[frame / 8] &= ~mask;
			}
		}
		else {
//			Tp* ptr = (Tp*)(&m_buffer.get()[m_frame_stride * node]);
			Tp* ptr = (Tp*)GetPtr(node);
			ptr[frame] = value;
		}
	}

	template <typename Tp>
	inline Tp Get(INDEX frame, INDEX node) const
	{
		if (typeid(Tp) == typeid(bool)) {
//			std::uint8_t* ptr = &(m_buffer.get()[m_frame_stride * node]);
			std::uint8_t* ptr = (std::uint8_t*)GetPtr(node);
			std::uint8_t mask = (std::uint8_t)(1 << (frame % 8));
			return ((ptr[frame / 8] & mask) != 0);
		}
		else {
//			Tp* ptr = (Tp*)(&m_buffer.get()[m_frame_stride * node]);
			Tp* ptr = (Tp*)GetPtr(node);
			return ptr[frame];
		}
	}


	void SetReal(INDEX frame, INDEX node, T value) const
	{
		switch (m_data_type) {
		case BB_TYPE_BINARY: Set<bool>(frame, node, value > (T)0.5);	break;
		case BB_TYPE_REAL32: Set<float>(frame, node, (float)value);		break;
		case BB_TYPE_REAL64: Set<double>(frame, node, (double)value);	break;
		}
	}

	T GetReal(INDEX frame, INDEX node) const
	{
		switch (m_data_type) {
		case BB_TYPE_BINARY: return Get<bool>(frame, node) ? (T)1.0 : (T)0.0;
		case BB_TYPE_REAL32: return (T)Get<float>(frame, node);
		case BB_TYPE_REAL64: return (T)Get<double>(frame, node);
		}
		return 0;
	}

	void SetBinary(INDEX frame, INDEX node, bool value) const
	{
		switch (m_data_type) {
		case BB_TYPE_BINARY: Set<bool>(frame, node, value);						break;
		case BB_TYPE_REAL32: Set<float>(frame, node, (value ? 1.0f : 0.0f)); 	break;
		case BB_TYPE_REAL64: Set<double>(frame, node, (value ? 1.0 : 0.0)); 	break;
		}
	}

	bool GetBinary(INDEX frame, INDEX node) const
	{
		switch (m_data_type) {
		case BB_TYPE_BINARY: return Get<bool>(frame, node);
		case BB_TYPE_REAL32: return (Get<float>(frame, node) > 0.5f);
		case BB_TYPE_REAL64: return (Get<double>(frame, node) > 0.5);
		}
		return false;
	}
};


}
