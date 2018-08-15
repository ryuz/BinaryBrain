

#pragma once

#include <vector>

// NeuralNetの抽象クラス
template <typename INDEX=size_t>
class NeuralNetLayer
{
public:
	virtual ~NeuralNetLayer() {}							// デストラクタ
	
	virtual void  SetInputValuePtr(const void* ptr) = 0;	// 入力側値アドレス設定
	virtual void  SetOutputValuePtr(void* ptr) = 0;			// 出力側値アドレス設定
	virtual void  SetOutputErrorPtr(const void* ptr) = 0;	// 出力側誤差アドレス設定
	virtual void  SetInputErrorPtr(void* ptr) = 0;			// 入力側誤差アドレス設定

	virtual INDEX GetInputFrameSize(void) const = 0;		// 入力のフレーム数
	virtual INDEX GetInputNodeSize(void) const = 0;			// 入力のノード数
	virtual INDEX GetOutputFrameSize(void) const = 0;		// 出力のフレーム数
	virtual INDEX GetOutputNodeSize(void) const = 0;		// 出力のノード数
	
	virtual int   GetInputValueBitSize(void) const = 0;		// 入力値のサイズ
	virtual int   GetInputErrorBitSize(void) const = 0;		// 出力値のサイズ
	virtual int   GetOutputValueBitSize(void) const = 0;	// 入力値のサイズ
	virtual int   GetOutputErrorBitSize(void) const = 0;	// 入力値のサイズ

	virtual	void  Forward(void) = 0;						// 予測
	virtual	void  Backward(void) = 0;						// 誤差逆伝播
	virtual	void  Update(double learning_rate) {};			// 学習
};

