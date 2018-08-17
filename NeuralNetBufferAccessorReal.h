

#pragma once


// NeuralNet用のバッファ
template <typename T = float, typename INDEX = size_t>
class NeuralNetBufferAccessorReal
{
protected:
	T*				m_buffer;
	INDEX			m_frame_size;
	
	T& at(INDEX frame, INDEX node) {
		return m_buffer[m_frame_size * node + frame];
	}
	
public:
	NeuralNetBufferAccessorReal(void* buffer, INDEX frame_size)
	{
		m_buffer     = (T*)buffer;
		m_frame_size = frame_size;
	}
	
	
	// native
	void Set(INDEX frame, INDEX node, T value)
	{
		at(frame, node) = value;
	}
	
	T Get(INDEX frame, INDEX node)
	{
		return at(frame, node);
	}
	
	
	// virtual
	virtual void SetReal(INDEX frame, INDEX node, T value)
	{
		Set(frame, node, value);
	}
	
	virtual T GetReal(INDEX frame, INDEX node)
	{
		return Get(frame, node);
	}
	
	virtual void SetBinary(INDEX frame, INDEX node, bool value)
	{
		Set(frame, node, value ? (T)1.0 : (T)0.0);
	}
	
	virtual bool GetBinary(INDEX frame, INDEX node)
	{
		return Get(frame, node) > (T)0.5 ? true : false;
	}
};


