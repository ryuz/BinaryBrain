#include <stdio.h>
#include <iostream>
#include "gtest/gtest.h"
#include "bb/OptimizerAdam.h"


class ModelAdam
{
public:
    float m_lr;
    float m_beta1;
    float m_beta2;
    float m_iter = 0;
    float m_m = 0;
    float m_v = 0;

    ModelAdam(float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f)
    {
        m_lr    = lr;
        m_beta1 = beta1;
        m_beta2 = beta2;
    }
    
    void update(float &params, float grads)
    {
        m_iter += 1;
        float lr_t = m_lr * sqrt(1.0f - pow(m_beta2, m_iter)) / (1.0f - pow(m_beta1, m_iter));

        m_m += (1 - m_beta1) * (grads - m_m);
        m_v += (1 - m_beta2) * (pow(grads, 2) - m_v);

//       std::cout << "[model] lr_t = " << lr_t << std::endl;
//       std::cout << "[model] m_m  = " << m_m << std::endl;
//       std::cout << "[model] m_v  = " << m_v << std::endl;
        
        params -= lr_t * m_m / (sqrt(m_v) + 1e-7f);

//        std::cout << "[model] lr_t * m_m / (sqrt(m_v) + 1e-7f)  = " << lr_t * m_m / (sqrt(m_v) + 1e-7f) << std::endl;
//        std::cout << "[model] sqrt(m_v) = " << sqrt(m_v) << std::endl;
    }
};


TEST(NeuralNetOptimizerAdamTest, testNeuralNetOptimizerAdamTest)
{
    float const learning_rate = 0.001f;
    float const beta1         = 0.9f;
    float const beta2         = 0.999f;

    ModelAdam   model(learning_rate, beta1, beta2);

    auto opt_adam = bb::OptimizerAdam<float>::Create(learning_rate, beta1, beta2);
    auto param_tensor = std::shared_ptr<bb::Tensor>(new bb::Tensor(BB_TYPE_FP32, 1));
    auto grad_tensor  = std::shared_ptr<bb::Tensor>(new bb::Tensor(BB_TYPE_FP32, 1));
    bb::Variables param_var;
    bb::Variables grad_var;
    param_var.PushBack(param_tensor);
    grad_var.PushBack(grad_tensor);
    opt_adam->SetVariables(param_var, grad_var);

    std::mt19937_64                 mt(1);
    std::normal_distribution<float> norm_dist(0.0001f, 2.0f);

    float exp_p = 0.1f;
    {
        auto param_ptr = param_tensor->GetPtr<float>();
        param_ptr(0) = exp_p;
    }

    for ( int i = 0; i < 10; i++ ) {
        {
            auto param_ptr = param_tensor->LockConst<float>();
        	EXPECT_FLOAT_EQ(exp_p, param_ptr(0));
        }

        float grad = norm_dist(mt);

        // 期待値作成
        model.update(exp_p, grad);
        
        // 計算
        {
            auto grad_ptr = grad_tensor->GetPtr<float>();
            grad_ptr(0) = grad;
        }
        opt_adam->Update();
        
        {
            auto param_ptr = param_tensor->LockConst<float>();
        	EXPECT_FLOAT_EQ(exp_p, param_ptr(0));
//          std::cout << grad << ", " << exp_p << ", " << param_ptr(0) << std::endl;
        }
    }
}


/*
TEST(NeuralNetOptimizerAdamTest, testNeuralNetOptimizerAdam2)
{
    int const n = 10;

    float const learning_rate = 0.001f;
    float const beta1         = 0.9f;
    float const beta2         = 0.999f;

    std::vector<ModelAdam>  models(n , ModelAdam(learning_rate, beta1, beta2));

    bb::NeuralNetOptimizerAdam<float>   opt_adam(learning_rate, beta1, beta2);
    auto opt = opt_adam.Create(n);

    std::mt19937_64                 mt(1);
    std::normal_distribution<float> norm_dist(0.0001f, 2.0f);

    std::vector<float> exp_p(n);
    std::vector<float> test_p(n);

    for ( int i = 0; i < n; ++ i ) {
        exp_p[i] = norm_dist(mt);
        test_p[i] = exp_p[i];
    }

    for ( int loop = 0; loop < 100; loop++ ) {
        std::vector<float> grad(n);
        for ( int i = 0; i < n; ++ i ) {
            grad[i] = norm_dist(mt);
        }

        for ( int i = 0; i < n; ++ i ) {
            models[i].update(exp_p[i], grad[i]);
        }

        opt->Update(test_p, grad);
        
        for ( int i = 0; i < n; ++ i ) {
        	EXPECT_FLOAT_EQ(exp_p[i], test_p[i]);
//          std::cout << loop << " " << i << " : " << exp_p[i] << ", " << test_p[i] << std::endl;
        }
    }
}

*/


TEST(OptimizerAdamTest, testOptimizerAdam_Cmp)
{
    auto param_cpu = std::shared_ptr< bb::Tensor >(new bb::Tensor(BB_TYPE_FP32, {11, 33}, true));
    auto param_gpu = std::shared_ptr< bb::Tensor >(new bb::Tensor(BB_TYPE_FP32, {11, 33}, false));
    auto grad_cpu  = std::shared_ptr< bb::Tensor >(new bb::Tensor(BB_TYPE_FP32, {11, 33}, true));
    auto grad_gpu  = std::shared_ptr< bb::Tensor >(new bb::Tensor(BB_TYPE_FP32, {11, 33}, false));

    std::mt19937_64 mt(1);
    std::normal_distribution<float> norm_dist(0.0f, 1.0f);

    bb::Variables param_var_cpu;
    bb::Variables param_var_gpu;
    bb::Variables grad_var_cpu;
    bb::Variables grad_var_gpu;

    param_var_cpu.PushBack(param_cpu);
    param_var_gpu.PushBack(param_gpu);
    grad_var_cpu.PushBack(grad_cpu);
    grad_var_gpu.PushBack(grad_gpu);

    auto opt_cpu = bb::OptimizerAdam<float>::Create();
    auto opt_gpu = bb::OptimizerAdam<float>::Create();

    opt_cpu->SetVariables(param_var_cpu, grad_var_cpu);
    opt_gpu->SetVariables(param_var_gpu, grad_var_gpu);

    {
        auto param_ptr_cpu = param_cpu->GetPtr<float>();
        auto param_ptr_gpu = param_gpu->GetPtr<float>();
        for ( bb::index_t i = 0; i < param_cpu->GetSize(); ++i ) {
            float param = norm_dist(mt);
            param_ptr_cpu[i] = param;
            param_ptr_gpu[i] = param;
        }
    }

    for ( int loop = 0; loop < 10; ++loop ) {
        {
            auto grad_ptr_cpu = grad_cpu->GetPtr<float>();
            auto grad_ptr_gpu = grad_gpu->GetPtr<float>();
            for ( bb::index_t i = 0; i < param_cpu->GetSize(); ++i ) {
                float grad = norm_dist(mt);
                grad_ptr_cpu[i] = grad;
                grad_ptr_gpu[i] = grad;
            }
        }

        {
            auto param_ptr_cpu = param_cpu->LockConst<float>();
            auto param_ptr_gpu = param_gpu->LockConst<float>();
            auto grad_ptr_cpu = grad_cpu->LockConst<float>();
            auto grad_ptr_gpu = grad_gpu->LockConst<float>();
            for ( bb::index_t i = 0; i < param_cpu->GetSize(); ++i ) {
            	EXPECT_FLOAT_EQ(param_ptr_cpu[i], param_ptr_gpu[i]);
            	EXPECT_FLOAT_EQ(grad_ptr_cpu[i], grad_ptr_gpu[i]);
            }
        }

        opt_cpu->Update();
        opt_gpu->Update();
    }
}


