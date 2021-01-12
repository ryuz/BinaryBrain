// --------------------------------------------------------------------------
//  BinaryBrain  -- binary network evaluation platform
//   MNIST sample
//
//                                Copyright (C) 2018-2019 by Ryuji Fuchikami
// --------------------------------------------------------------------------


#include <iostream>
#include <fstream>
#include <numeric>
#include <random>
#include <chrono>

#include "bb/DenseAffine.h"
#include "bb/BatchNormalization.h"
#include "bb/ReLU.h"
#include "bb/LossSoftmaxCrossEntropy.h"
#include "bb/MetricsCategoricalAccuracy.h"
#include "bb/OptimizerAdam.h"
#include "bb/OptimizerSgd.h"
#include "bb/LoadMnist.h"
#include "bb/ShuffleSet.h"
#include "bb/Utility.h"



// 自分でモデル構築する例(bb::Modelを継承すること)
class MnistMyCustomModel : public bb::Model
{
protected:
    using Affine   = bb::DenseAffine<float>;
    using Activate = bb::ReLU<float>;

public:
    // 既存クラスは shared_ptr で生成される
    std::shared_ptr<Affine>   m_affine0;
    std::shared_ptr<Activate> m_activate0;
    std::shared_ptr<Affine>   m_affine1;
    std::shared_ptr<Activate> m_activate1;
    std::shared_ptr<Affine>   m_affine2;

public:
    // コンストラクタ
    MnistMyCustomModel()
    {
        m_affine0   = Affine::Create(256);
        m_activate0 = Activate::Create();
        m_affine1   = Affine::Create(512);
        m_activate1 = Activate::Create();
        m_affine2   = Affine::Create(10);
    }

    // モデル名を返す(シリアライズ時のオブジェクトIDとして利用)
    std::string GetModelName(void) const override
    {
        return "MnistMyCustomModel";
    }

    // SendCommandを定義すればコマンドの受け取りや子への伝搬も可能(必須ではない)
    void SendCommand(std::string command, std::string send_to = "all") override
    {
        m_affine0->SendCommand(command, send_to);
        m_activate0->SendCommand(command, send_to);
        m_affine1->SendCommand(command, send_to);
        m_activate1->SendCommand(command, send_to);
        m_affine2->SendCommand(command, send_to);
    }

    // 入力シェイプを設定し、出力シェイプを返す(必須)
    bb::indices_t SetInputShape(bb::indices_t shape) override
    {
        shape = m_affine0->SetInputShape(shape);
        shape = m_activate0->SetInputShape(shape);
        shape = m_affine1->SetInputShape(shape);
        shape = m_activate1->SetInputShape(shape);
        shape = m_affine2->SetInputShape(shape);
        return shape;
    }

    // 入力シェイプを取得(必須)
    bb::indices_t GetInputShape(void) const override
    {
        return m_affine0->GetInputShape();
    }

    // 出力シェイプを取得(必須)
    bb::indices_t GetOutputShape(void) const override
    {
        return m_affine2->GetInputShape();
    }

    // パラメータを返す(学習する場合は必須)
    bb::Variables GetParameters(void) override
    {
        bb::Variables var;
        var.PushBack(m_affine0->GetParameters());
        var.PushBack(m_activate0->GetParameters());
        var.PushBack(m_affine1->GetParameters());
        var.PushBack(m_activate1->GetParameters());
        var.PushBack(m_affine2->GetParameters());
        return var;
    }

    // 勾配を返す(学習する場合は必須)
    bb::Variables GetGradients(void) override
    {
        bb::Variables var;
        var.PushBack(m_affine0->GetGradients());
        var.PushBack(m_activate0->GetGradients());
        var.PushBack(m_affine1->GetGradients());
        var.PushBack(m_activate1->GetGradients());
        var.PushBack(m_affine2->GetGradients());
        return var;
    }

    // forward計算(必須)
    bb::FrameBuffer Forward(bb::FrameBuffer x, bool train=true) override
    {
        x = m_affine0->Forward(x, train);
        x = m_activate0->Forward(x, train);
        x = m_affine1->Forward(x, train);
        x = m_activate1->Forward(x, train);
        x = m_affine2->Forward(x, train);
        return x;
    }

    // backword計算(学習する場合は必須)
    bb::FrameBuffer Backward(bb::FrameBuffer dy) override
    {
        dy = m_affine2->Backward(dy);
        dy = m_activate1->Backward(dy);
        dy = m_affine1->Backward(dy);
        dy = m_activate0->Backward(dy);
        dy = m_affine0->Backward(dy);
        return dy;
    }

protected:
    // 保存用シリアライズ(DumpObjectから呼ばれる)
    void DumpObjectData(std::ostream &os) const override
    {
        m_affine0->DumpObject(os);
        m_activate0->DumpObject(os);
        m_affine1->DumpObject(os);
        m_activate1->DumpObject(os);
        m_affine2->DumpObject(os);
    }

    // 復帰用シリアライズ(LoadObjectから呼ばれる)
    void LoadObjectData(std::istream &is) override
    {
        m_affine0->LoadObject(is);
        m_activate0->LoadObject(is);
        m_affine1->LoadObject(is);
        m_activate1->LoadObject(is);
        m_affine2->LoadObject(is);
    }
};



// カスタムモデル
void MnistCustomModel(int epoch_size, int mini_batch_size, bool binary_mode, bool file_read)
{
    // データセット準備(MNIST)
    std::mt19937_64 mt(1);  // シャッフル用の乱数
#ifdef _DEBUG
    auto td = bb::LoadMnist<>::Load(512, 128);
    std::cout << "!!! debug mode !!!" << std::endl;
#else
    auto td = bb::LoadMnist<>::Load();
#endif
    
    // ネット構築
    MnistMyCustomModel  net;
    net.SetInputShape(td.x_shape);

    // 前の学習を読み込む
    if (file_read) {
        net.LoadFromFile("MnistCustomModel.bb_net");
    }

    // Learning
    auto lossFunc    = bb::LossSoftmaxCrossEntropy<float>::Create();
    auto metricsFunc = bb::MetricsCategoricalAccuracy<float>::Create();
    
    auto optimizer = bb::OptimizerAdam<float>::Create();
    optimizer->SetVariables(net.GetParameters(), net.GetGradients());

    // モード設定
    if ( binary_mode ) {
        net.SendCommand("binary true");
    }
    else {
        net.SendCommand("binary false");
    }
    
    // 学習実施
    bb::FrameBuffer x(mini_batch_size, {28, 28, 1}, BB_TYPE_FP32);
    bb::FrameBuffer t(mini_batch_size, {10},        BB_TYPE_FP32);
    for ( bb::index_t epoch = 0; epoch < epoch_size; ++epoch ) {
        std::cout << "epoch : " << epoch << std::endl;

        // training
        lossFunc->Clear();
        metricsFunc->Clear();
        for (bb::index_t i = 0; i < (bb::index_t)(td.x_train.size() - mini_batch_size); i += mini_batch_size)
        {
            // 学習データセット
            x.SetVector(td.x_train, i);
            t.SetVector(td.t_train, i);
            
            // 学習実施
            auto y = net.Forward(x);
            auto dy = lossFunc->CalculateLoss(y, t, y.GetFrameSize());
            metricsFunc->CalculateMetrics(y, t);
            dy = net.Backward(dy);
            optimizer->Update();
        }
        std::cout << "[train] accuracy : " << metricsFunc->GetMetrics() << "  loss : " << lossFunc->GetLoss() << std::endl;
        
        // test
        lossFunc->Clear();
        metricsFunc->Clear();
        for (bb::index_t i = 0; i < (bb::index_t)(td.x_test.size() - mini_batch_size); i += mini_batch_size)
        {
            // テストデータセット
            x.SetVector(td.x_test, i);
            t.SetVector(td.t_test, i);
            
            // 評価実施
            auto y = net.Forward(x);
            auto dy = lossFunc->CalculateLoss(y, t, y.GetFrameSize());
            metricsFunc->CalculateMetrics(y, t);
        }
        std::cout << "[test]  accuracy : " << metricsFunc->GetMetrics() << "  loss : " << lossFunc->GetLoss() << std::endl;

        // ファイルに保存
        net.DumpToFile("MnistCustomModel.bb_net");

        // 学習データをシャッフル
        bb::ShuffleDataSet(mt(), td.x_train, td.t_train);
    }
}


// end of file
