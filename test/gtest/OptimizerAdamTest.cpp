#include <stdio.h>
#include <iostream>
#include "gtest/gtest.h"

#include "bb/OptimizerAdam.h"



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

    bb::OptimizerAdam<float> opt_cpu;
    bb::OptimizerAdam<float> opt_gpu;

    opt_cpu.SetVariables(param_var_cpu, grad_var_cpu);
    opt_gpu.SetVariables(param_var_gpu, grad_var_gpu);

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
            auto param_ptr_cpu = param_cpu->GetConstPtr<float>();
            auto param_ptr_gpu = param_gpu->GetConstPtr<float>();
            auto grad_ptr_cpu = grad_cpu->GetConstPtr<float>();
            auto grad_ptr_gpu = grad_gpu->GetConstPtr<float>();
            for ( bb::index_t i = 0; i < param_cpu->GetSize(); ++i ) {
            	EXPECT_FLOAT_EQ(param_ptr_cpu[i], param_ptr_gpu[i]);
            	EXPECT_FLOAT_EQ(grad_ptr_cpu[i], grad_ptr_gpu[i]);
            }
        }

        opt_cpu.Update();
        opt_gpu.Update();
    }
}

