#include <stdio.h>
#include <iostream>
#include "gtest/gtest.h"
#include "bb/NeuralNetLayer.h"
#include "bb/NeuralNetOptimizerAdam.h"


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
        
        params -= lr_t * m_m / (sqrt(m_v) + 1e-7f);
    }
};


TEST(NeuralNetOptimizerAdamTest, testNeuralNetOptimizerAdamTest)
{
    float const learning_rate = 0.001f;
    float const beta1         = 0.9f;
    float const beta2         = 0.999f;

    ModelAdam   model(learning_rate, beta1, beta2);

    bb::NeuralNetOptimizerAdam<float>   opt_adam(learning_rate, beta1, beta2);
    auto opt = opt_adam.Create(1);

    std::mt19937_64                 mt(1);
    std::normal_distribution<float> norm_dist(0.0001f, 2.0f);

    float exp_p = 0.1f;
    float test_p = exp_p;
    for ( int i = 0; i < 100; i++ ) {
        float grad = norm_dist(mt);

        model.update(exp_p, grad);

        opt->Update(test_p, grad);
        
    	EXPECT_FLOAT_EQ(exp_p, test_p);
//      std::cout << exp_p << ", " << test_p << std::endl;
    }
}



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

