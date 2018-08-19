

#pragma once

#include <intrin.h>
#include "NeuralNetBufferAccessor.h"


// NeuralNet用のバッファ
template <typename T = float, typename INDEX = size_t>
class NeuralNetBufferAccessorBinary : public NeuralNetBufferAccessor<T, INDEX>
{
protected:
	std::uint8_t*	m_buffer;
	INDEX			m_frame_size;
	
	std::uint8_t& at(INDEX frame, INDEX node) {
		return m_buffer[m_frame_size * node + (frame / 8)];
	}
	
	std::uint8_t mask_bit(INDEX frame, INDEX node) {
		return (1 << (frame % 8));
	}

public:
	NeuralNetBufferAccessorBinary(void* buffer, INDEX frame_size)
	{
		m_buffer     = (std::uint8_t*)buffer;
		m_frame_size = ((frame_size + 255) / 256) * 32;
	}

	NeuralNetBufferAccessorBinary(const NeuralNetBufferAccessorBinary& acc)
	{
		m_buffer = acc.m_buffer;
		m_frame_size = acc.m_frame_size;
	}

	// clone
//	NeuralNetBufferAccessorr<T, INDEX>* clone(void) const
//	{
//		return new NeuralNetBufferAccessorBinary<T, INDEX>(*this);
//	}
	virtual NeuralNetBufferAccessor<T, INDEX>* clone(void) const
	{
		return new NeuralNetBufferAccessorBinary<T, INDEX>(*this);
//		return nullptr;
	}

	// native
	void Set(INDEX frame, INDEX node, bool value)
	{
		if ( value ) {
			at(frame, node) |= mask_bit(frame, node);
		}
		else {
			at(frame, node) &= ~mask_bit(frame, node);
		}
	}
	
	bool Get(INDEX frame, INDEX node)
	{
		return (at(frame, node) & mask_bit(frame, node)) != 0;
	}
	
	__m256i* GetMm256iPtr(INDEX node)
	{
		return (__m256i *)&m_buffer[m_frame_size * node];
	}
	
	// virtual
	virtual void SetReal(INDEX frame, INDEX node, T value)
	{
		Set(frame, node, value > (T)0.5);
	}
	
	virtual T GetReal(INDEX frame, INDEX node)
	{
		return Get(frame, node) ? (T)1.0 : (T)0.0;
	}
	
	virtual void SetBinary(INDEX frame, INDEX node, bool value)
	{
		Set(frame, node, value);
	}
	
	bool GetBinary(INDEX frame, INDEX node)
	{
		return Get(frame, node);
	}
	
};


