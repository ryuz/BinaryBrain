// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once


#include "bb/NeuralNetLayerBuf.h"


namespace bb {

// NeuralNetFilter2dクラス
template <typename T = float>
class NeuralNetFilter2d : public NeuralNetLayerBuf<T>
{
public:
	virtual int GetInputChannel(void)  const { return 1; }
	virtual int GetInputHeight(void)   const { return 1; }
	virtual int GetInputWidth(void)    const { return (int)this->GetInputNodeSize(); }
	virtual int GetFilterHeight(void)  const { return 1; }
	virtual int GetFilterWidth(void)   const { return 1; }
	virtual int GetOutputChannel(void) const { return 1; }
	virtual int GetOutputHeight(void)  const { return 1; }
	virtual int GetOutputWidth(void)   const { return (int)this->GetOutputNodeSize(); }
};


}