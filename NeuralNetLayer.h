


#pragma once


#include <vector>


// NeuralNetの抽象クラス
template <typename T=float, typename ET = float, typename INDEX=size_t>
class NeuralNetLayer
{
public:
	virtual ~NeuralNetLayer() {}								// デストラクタ
	
	// 基本部分
	virtual INDEX GetInputFrameSize(void) const = 0;	// 入力のフレーム数
	virtual INDEX GetInputNodeSize(void) const = 0;		// 入力のノード数
	virtual INDEX GetOutputFrameSize(void) const = 0;	// 出力のフレーム数
	virtual INDEX GetOutputNodeSize(void) const = 0;	// 出力のノード数
	
	virtual	void  Forward(void) = 0;
	virtual	void  Backward(void) = 0;
	
};

