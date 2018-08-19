

#pragma once

#include <memory>
#include <malloc.h>
#include "NeuralNetBufferAccessor.h"
#include "NeuralNetBufferAccessorReal.h"
#include "NeuralNetBufferAccessorBinary.h"


// NeuralNet用のバッファアクセサ
template <typename T = float, typename INDEX = size_t>
class NeuralNetBuffer
{
protected:
	std::shared_ptr<void>									m_buffer;
	INDEX													m_stride;

	std::unique_ptr< NeuralNetBufferAccessor<T, INDEX> >	m_accessor;
	INDEX m_frame_size = 0;
	INDEX m_node_size = 0;
	int   m_type_bit_size = 0;
	
	size_t CalcBufferSize(INDEX frame_size, INDEX node_size, int bit_size)
	{
		size_t mm256_size = ((frame_size * bit_size) + 255) / 256;
		return 32 * mm256_size * node_size;
	}

public:
	NeuralNetBuffer() {}
	NeuralNetBuffer(const NeuralNetBuffer& buf)
	{
		*this = buf;
	}

	NeuralNetBuffer(INDEX frame_size, INDEX node_size, int type_bit_size, int type = 0)
	{
		Resize(frame_size, node_size, type_bit_size, type);
	}

	NeuralNetBuffer& operator=(const NeuralNetBuffer &buf)
	{
		m_buffer = buf.m_buffer;
		m_accessor = std::unique_ptr< NeuralNetBufferAccessor<T, INDEX> >(buf.m_accessor->clone());
		m_frame_size = buf.m_frame_size;
		m_node_size = buf.m_node_size;
		m_type_bit_size = buf.m_type_bit_size;
		return *this;
	}

	void Resize(INDEX frame_size, INDEX node_size, int type_bit_size, int type = 0)
	{
		// メモリ確保
		size_t mem_size = CalcBufferSize(frame_size, node_size, type_bit_size);
		m_buffer = std::shared_ptr<void>(_aligned_malloc(mem_size, 32), _aligned_free);

		// アクセサ生成
		if (type_bit_size == 1) {
			m_accessor = std::make_unique< NeuralNetBufferAccessorBinary<T, INDEX> >(m_buffer.get(), frame_size);
		}
		else {
			m_accessor = std::make_unique< NeuralNetBufferAccessorReal<T, INDEX> >(m_buffer.get(), frame_size);
		}
	}

	INDEX GetFrameSize(void) { return m_frame_size; }
	INDEX GetNodeSize(void) { return m_node_size; }
	INDEX GetTypeBitSize(void) { return m_type_bit_size; }

	void* GetBufferPtr(void)
	{
		return m_buffer.get();
	}

	NeuralNetBufferAccessor<T, INDEX>* GetAccessor(void)
	{
		return m_accessor.get();
	}
};


