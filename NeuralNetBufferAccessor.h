

#pragma once


// NeuralNet用のバッファアクセサ
template <typename T = float, typename INDEX = size_t>
class NeuralNetBufferAccessor
{
public:
	virtual NeuralNetBufferAccessor<T, INDEX>* clone(void) const = 0;

	virtual void SetReal(INDEX frame, INDEX node, T value) = 0;
	virtual T    GetReal(INDEX frame, INDEX node) = 0;
	virtual void SetBinary(INDEX frame, INDEX node, bool value) = 0;
	virtual bool GetBinary(INDEX frame, INDEX node) = 0;
};


