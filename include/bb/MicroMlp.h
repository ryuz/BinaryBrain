// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once

#include <cstdint>
#include <random>

#include "bb/Model.h"
#include "bb/MicroMlpAffine.h"
#include "bb/BatchNormalization.h"
#include "bb/ReLU.h"


namespace bb {


// Sparce Mini-MLP(Multilayer perceptron) Layer [Affine-ReLU-Affine-BatchNorm-Binarize]
template <int N = 6, int M = 16, typename T = float, class Activation = ReLU<T> >
class MicroMlp : public Model
{
protected:
	// 3層で構成
	std::shared_ptr< MicroMlpAffine<N, M, T> >	m_affine;
	std::shared_ptr< BatchNormalization<T>   >  m_batch_norm;
	std::shared_ptr< Activation              >  m_activation;

protected:
	MicroMlp() {}

public:
	~MicroMlp() {}

    /*
    struct create_t
    {
        MicroMlpAffine<N, M, T>::create_t   affine;
        BatchNormalization<T>::create_t     bn;
    };

    static std::shared_ptr< MicroMlp > Create(create_t const &create)
    {
        auto self = std::shared_ptr<MicroMlp>(new MicroMlp);
        self->m_affine     = MicroMlpAffine<N, M, T>::Create(create.affine);
        self->m_batch_norm = BatchNormalization<T>::Create(create.bn);
        self->m_activation = Activation::Create();
        return self;
    }
    */

    static std::shared_ptr< MicroMlp > Create(index_t output_node_size, T momentum = (T)0.001)
    {
        auto self = std::shared_ptr<MicroMlp>(new MicroMlp);
        self->m_affine     = MicroMlpAffine<N, M, T>::Create(output_node_size);
        self->m_batch_norm = BatchNormalization<T>::Create(momentum);
        self->m_activation = Activation::Create();
        return self;
    }

	std::string GetClassName(void) const { return "MicroMlp"; }

    /*
	std::vector<T> CalcNode(INDEX node, std::vector<T> input_value) const
	{
		auto vec0 = m_affine.CalcNode(node, input_value);
		auto vec1 = m_batch_norm.CalcNode(node, vec0);
		auto vec2 = m_activation.CalcNode(node, vec1);
		return vec2;
	}
    */

    /**
     * @brief  コマンドを送る
     * @detail コマンドを送る
     */   
    void SendCommand(std::string command, std::string send_to = "all")
    {
	    m_affine    ->SendCommand(command, send_to);
	    m_batch_norm->SendCommand(command, send_to);
	    m_activation->SendCommand(command, send_to);
    }
    
    /**
     * @brief  パラメータ取得
     * @detail パラメータを取得する
     *         Optimizerでの利用を想定
     * @return パラメータを返す
     */
    Variables GetParameters(void)
    {
        Variables parameters;
	    parameters.PushBack(m_affine    ->GetParameters());
	    parameters.PushBack(m_batch_norm->GetParameters());
	    parameters.PushBack(m_activation->GetParameters());
        return parameters;
    }

    /**
     * @brief  勾配取得
     * @detail 勾配を取得する
     *         Optimizerでの利用を想定
     * @return パラメータを返す
     */
    virtual Variables GetGradients(void)
    {
        Variables gradients;
	    gradients.PushBack(m_affine    ->GetGradients());
	    gradients.PushBack(m_batch_norm->GetGradients());
	    gradients.PushBack(m_activation->GetGradients());
        return gradients;
    }  

    /**
     * @brief  入力形状設定
     * @detail 入力形状を設定する
     *         内部変数を初期化し、以降、GetOutputShape()で値取得可能となることとする
     *         同一形状を指定しても内部変数は初期化されるものとする
     * @param  shape      1フレームのノードを構成するshape
     * @return 出力形状を返す
     */
    indices_t SetInputShape(indices_t shape)
    {
	    shape = m_affine    ->SetInputShape(shape);
	    shape = m_batch_norm->SetInputShape(shape);
	    shape = m_activation->SetInputShape(shape);
        return shape;
    }

   /**
     * @brief  forward演算
     * @detail forward演算を行う
     * @param  x     入力データ
     * @param  train 学習時にtrueを指定
     * @return forward演算結果
     */
    FrameBuffer Forward(FrameBuffer x, bool train = true)
    {
	    x = m_affine    ->Forward(x, train);
	    x = m_batch_norm->Forward(x, train);
	    x = m_activation->Forward(x, train);
        return x;
    }

   /**
     * @brief  backward演算
     * @detail backward演算を行う
     *         
     * @return backward演算結果
     */
    FrameBuffer Backward(FrameBuffer dy)
    {
	    dy = m_activation->Backward(dy);
	    dy = m_batch_norm->Backward(dy);
	    dy = m_affine    ->Backward(dy);
        return dy; 
    }
	
public:
    /*
	void Save(cereal::JSONOutputArchive& archive) const
	{
	    m_affine    ->Save(archive);
	    m_batch_norm->Save(archive);
	    m_activation->Save(archive);
	}

	void Load(cereal::JSONInputArchive& archive)
	{
	    m_affine    ->Load(archive);
	    m_batch_norm->Load(archive);
	    m_activation->Load(archive);
	}
    */

};


}