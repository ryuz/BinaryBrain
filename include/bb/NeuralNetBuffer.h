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
	INDEX							m_stride = 0;

	int								m_data_type = 0;
	INDEX							m_frame_size = 0;
	INDEX							m_node_size = 0;

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
		m_stride = buf.m_stride;

		m_data_type = buf.m_data_type;
		m_frame_size = buf.m_frame_size;
		m_node_size = buf.m_node_size;

		return *this;
	}

	void Resize(INDEX frame_size, INDEX node_size, int data_type)
	{
		// 設定保存
		m_data_type = data_type;
		m_frame_size = frame_size;
		m_node_size = node_size;

		size_t type_bit_size = NeuralNet_GetTypeBitSize(data_type);

		// メモリ確保
		if (data_type == BB_TYPE_BINARY) {
			m_stride = (((frame_size * type_bit_size) + 255) / 256) * 32;
		}
		else {
			m_stride = frame_size * type_bit_size / 8;
		}
		m_buffer = std::shared_ptr<std::uint8_t>((std::uint8_t *)_aligned_malloc(m_stride*m_node_size, 32), _aligned_free);
	}

	INDEX GetFrameSize(void) { return m_frame_size; }
	INDEX GetNodeSize(void) { return m_node_size; }
	INDEX GetTypeBitSize(void) { return m_type_bit_size; }

	void* GetBuffer(void)
	{
		return m_buffer.get();
	}

	template <typename Tp>
	inline Tp* GetPtr(INDEX node)
	{
		return (Tp*)(&m_buffer.get()[m_stride * node]);
	}

	template <typename Tp>
	inline void Set(INDEX frame, INDEX node, Tp value)
	{
		if (typeid(Tp) == typeid(bool)) {
			std::uint8_t* ptr = &(m_buffer.get()[m_stride * node]);
			std::uint8_t mask = (std::uint8_t)(1 << (frame % 8));
			if (value) {
				ptr[frame / 8] |= mask;
			}
			else {
				ptr[frame / 8] &= ~mask;
			}
		}
		else {
			Tp* ptr = (Tp*)(&m_buffer.get()[m_stride * node]);
			ptr[frame] = value;
		}
	}

	template <typename Tp>
	inline Tp Get(INDEX frame, INDEX node)
	{
		if (typeid(Tp) == typeid(bool)) {
			std::uint8_t* ptr = &(m_buffer.get()[m_stride * node]);
			std::uint8_t mask = (std::uint8_t)(1 << (frame % 8));
			return ((ptr[frame / 8] & mask) != 0);
		}
		else {
			Tp* ptr = (Tp*)(&m_buffer.get()[m_stride * node]);
			return ptr[frame];
		}
	}


	void SetReal(INDEX frame, INDEX node, T value)
	{
		switch (m_data_type) {
		case BB_TYPE_BINARY: Set<bool>(frame, node, value > (T)0.5);	break;
		case BB_TYPE_REAL32: Set<float>(frame, node, (float)value);		break;
		case BB_TYPE_REAL64: Set<double>(frame, node, (double)value);	break;
		}
	}

	T GetReal(INDEX frame, INDEX node)
	{
		switch (m_data_type) {
		case BB_TYPE_BINARY: return Get<bool>(frame, node) ? (T)1.0 : (T)0.0;
		case BB_TYPE_REAL32: return (T)Get<float>(frame, node);
		case BB_TYPE_REAL64: return (T)Get<double>(frame, node);
		}
		return 0;
	}

	void SetBinary(INDEX frame, INDEX node, bool value)
	{
		switch (m_data_type) {
		case BB_TYPE_BINARY: Set<bool>(frame, node, value);						break;
		case BB_TYPE_REAL32: Set<float>(frame, node, (value ? 1.0f : 0.0f)); 	break;
		case BB_TYPE_REAL64: Set<double>(frame, node, (value ? 1.0 : 0.0)); 	break;
		}
	}

	bool GetBinary(INDEX frame, INDEX node)
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
