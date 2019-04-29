#include <stdio.h>
#include <iostream>
#include <chrono>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include "bb/ConvolutionIm2Col.h"
#include "bb/ConvolutionCol2Im.h"

#include "bbcu/bbcu.h"
#include "bbcu/bbcu_util.h"



CUBB_DLL_EXPORT int bbcu_eva_Im2Col_Forward
        (
            const float*    input_sig_buf,
            int             input_frame_size,
            int             input_frame_stride,
            int             input_w_size,
            int             input_h_size,
            int             input_c_size,
            
            float*          output_sig_buf,
            int             output_frame_stride,

            int             filter_w_size,
            int             filter_h_size
        )
{
    cudaDeviceProp dev;
    BB_CUDA_SAFE_CALL(cudaGetDeviceProperties(&dev, 0));

    cudaError_t cudaStatus0 = cudaGetLastError();
    if (cudaStatus0 != cudaSuccess) {
        fprintf(stderr, "start failed: %s\n", cudaGetErrorString(cudaStatus0));
        exit(1);
    }

    cudaDeviceSynchronize();
    auto time0 = std::chrono::system_clock::now();

    int output_c_size = input_c_size;
    int output_w_size = input_w_size - filter_w_size + 1;
    int output_h_size = input_h_size - filter_h_size + 1;
    int output_size   = output_w_size * output_h_size;

    int input_node_size  = input_c_size * input_h_size * input_w_size;
    int output_node_size = output_c_size * filter_h_size * filter_w_size;
    
    int output_frame_size = input_frame_size * output_size;
    
    float* input_sig_dev_buf;
    float* output_sig_dev_buf;

    BB_CUDA_SAFE_CALL(cudaMalloc((void**)&input_sig_dev_buf,  input_node_size  * input_frame_stride * sizeof(float)));
    BB_CUDA_SAFE_CALL(cudaMalloc((void**)&output_sig_dev_buf, output_node_size * output_frame_stride * sizeof(float)));
    
    cudaDeviceSynchronize();
    auto time1 = std::chrono::system_clock::now();

    BB_CUDA_SAFE_CALL(cudaMemcpy(input_sig_dev_buf, input_sig_buf, input_node_size * input_frame_stride * sizeof(float), cudaMemcpyHostToDevice));

    cudaDeviceSynchronize();
    auto time2 = std::chrono::system_clock::now();
    
    cubb_Im2Col_Forward(
            input_sig_dev_buf,
            input_frame_size,
            input_frame_stride,
            input_w_size,
            input_h_size,
            input_c_size,
            
            output_sig_dev_buf,
            output_frame_stride,

            filter_w_size,
            filter_h_size
        );
    
    /*
    int     frame_unit = 16;
    dim3    grid(output_frame_size/frame_unit, output_c_size);
    dim3    block(frame_unit, filter_w_size, filter_h_size);
    
    kernal_Im2Col_Forward<<<grid, block>>>(
            input_sig_dev_buf,
            input_frame_stride,
            input_w_size,
            input_h_size,
            
            output_sig_dev_buf,
            output_frame_size,
            output_frame_stride,
            output_w_size,
            output_size
        );
    BB_CUDA_CHECK_LAST_ERROR();
    */

    cudaDeviceSynchronize();
    auto time3 = std::chrono::system_clock::now();

    BB_CUDA_SAFE_CALL(cudaMemcpy(output_sig_buf, output_sig_dev_buf, output_node_size * output_frame_stride * sizeof(float), cudaMemcpyDeviceToHost));

    cudaDeviceSynchronize();
    auto time4 = std::chrono::system_clock::now();

    BB_CUDA_SAFE_CALL(cudaFree(input_sig_dev_buf));
    BB_CUDA_SAFE_CALL(cudaFree(output_sig_dev_buf));

    cudaDeviceSynchronize();
    auto time5 = std::chrono::system_clock::now();

    double elapsed_malloc       = (double)std::chrono::duration_cast<std::chrono::milliseconds>(time1-time0).count();
    double elapsed_cpu_to_gpu   = (double)std::chrono::duration_cast<std::chrono::milliseconds>(time2-time1).count();
    double elapsed_kernel       = (double)std::chrono::duration_cast<std::chrono::milliseconds>(time3-time2).count();
    double elapsed_gpu_to_cpu   = (double)std::chrono::duration_cast<std::chrono::milliseconds>(time4-time3).count();
    double elapsed_free         = (double)std::chrono::duration_cast<std::chrono::milliseconds>(time5-time4).count();
    std::cout << "malloc               : " << elapsed_malloc       << " [msec]" << std::endl;
    std::cout << "data copy(cpu->gpu)  : " << elapsed_cpu_to_gpu   << " [msec]" << std::endl;
    std::cout << "kernel               : " << elapsed_kernel       << " [msec]" << std::endl;
    std::cout << "data copy(gpu->cpu)  : " << elapsed_gpu_to_cpu   << " [msec]" << std::endl;
    std::cout << "free                 : " << elapsed_free         << " [msec]" << std::endl;
    
    return 0;
}




CUBB_DLL_EXPORT int bbcu_eva_Im2Col_Backward
        (
            float*          input_grad_buf,
            int             input_frame_size,
            int             input_frame_stride,
            int             input_w_size,
            int             input_h_size,
            int             input_c_size,
            const float*    output_grad_buf,
            int             output_frame_stride,
            int             filter_w_size,
            int             filter_h_size
        )
{
    cudaDeviceSynchronize();
    auto time0 = std::chrono::system_clock::now();

    int output_c_size = input_c_size;
    int output_w_size = input_w_size - filter_w_size + 1;
    int output_h_size = input_h_size - filter_h_size + 1;
    int output_size   = output_w_size * output_h_size;

    int input_node_size  = input_c_size * input_h_size * input_w_size;
    int output_node_size = output_c_size * filter_h_size * filter_w_size;
    
    int output_frame_size = input_frame_size * output_size;
    
    float* input_grad_dev_buf;
    float* output_grad_dev_buf;

    BB_CUDA_SAFE_CALL(cudaMalloc((void**)&input_grad_dev_buf,   input_node_size  * input_frame_size * sizeof(float)));
    BB_CUDA_SAFE_CALL(cudaMalloc((void**)&output_grad_dev_buf,  output_node_size * output_frame_size * sizeof(float)));
    
    cudaDeviceSynchronize();
    auto time1 = std::chrono::system_clock::now();

    BB_CUDA_SAFE_CALL(cudaMemcpy(output_grad_dev_buf, output_grad_buf, output_node_size * output_frame_stride * sizeof(float), cudaMemcpyHostToDevice));

    cudaDeviceSynchronize();
    auto time2 = std::chrono::system_clock::now();
    
    cubb_Im2Col_Backward(
            input_grad_dev_buf,
            input_frame_size,
            input_frame_stride,
            input_w_size,
            input_h_size,
            input_c_size,
            output_grad_dev_buf,
            output_frame_stride,
            filter_w_size,
            filter_h_size
        );
    
    cudaDeviceSynchronize();
    auto time3 = std::chrono::system_clock::now();

    BB_CUDA_SAFE_CALL(cudaMemcpy(input_grad_buf, input_grad_dev_buf, input_node_size * input_frame_stride * sizeof(float), cudaMemcpyDeviceToHost));

    cudaDeviceSynchronize();
    auto time4 = std::chrono::system_clock::now();

    BB_CUDA_SAFE_CALL(cudaFree(input_grad_buf));
    BB_CUDA_SAFE_CALL(cudaFree(output_grad_dev_buf));

    cudaDeviceSynchronize();
    auto time5 = std::chrono::system_clock::now();

    double elapsed_malloc       = (double)std::chrono::duration_cast<std::chrono::milliseconds>(time1-time0).count();
    double elapsed_cpu_to_gpu   = (double)std::chrono::duration_cast<std::chrono::milliseconds>(time2-time1).count();
    double elapsed_kernel       = (double)std::chrono::duration_cast<std::chrono::milliseconds>(time3-time2).count();
    double elapsed_gpu_to_cpu   = (double)std::chrono::duration_cast<std::chrono::milliseconds>(time4-time3).count();
    double elapsed_free         = (double)std::chrono::duration_cast<std::chrono::milliseconds>(time5-time4).count();
    std::cout << "malloc               : " << elapsed_malloc       << " [msec]" << std::endl;
    std::cout << "data copy(cpu->gpu)  : " << elapsed_cpu_to_gpu   << " [msec]" << std::endl;
    std::cout << "kernel               : " << elapsed_kernel       << " [msec]" << std::endl;
    std::cout << "data copy(gpu->cpu)  : " << elapsed_gpu_to_cpu   << " [msec]" << std::endl;
    std::cout << "free                 : " << elapsed_free         << " [msec]" << std::endl;
    
    return 0;
}





#if 0

inline void testSetupLayerBuffer(bb::NeuralNetLayer<>& net)
{
    net.SetInputSignalBuffer (net.CreateInputSignalBuffer());
    net.SetInputErrorBuffer (net.CreateInputErrorBuffer());
    net.SetOutputSignalBuffer(net.CreateOutputSignalBuffer());
    net.SetOutputErrorBuffer(net.CreateOutputErrorBuffer());
}


TEST(cudaIm2ColTest, testcudaIm2Col)
{
    cudaDeviceProp dev;
    cudaGetDeviceProperties(&dev, 0);
    cudaDeviceSynchronize();

//  float* ptr_buf = 0;
////    cudaMallocManaged((void**)&ptr_buf, 64*1024*1024);
//  cudaMalloc((void**)&ptr_buf, 64*1024*1024);
//  cudaFree(ptr_buf);
    
#if _DEBUG
    const int input_frame_size = 16;
    const int c_size = 4;
    const int input_h_size = 28;
    const int input_w_size = 28;
    const int filter_h_size = 3;
    const int filter_w_size = 3;
#else
    const int input_frame_size = 256;
    const int c_size = 32;
    const int input_h_size = 28;
    const int input_w_size = 28;
    const int filter_h_size = 3;
    const int filter_w_size = 3;
#endif

    int output_w_size = input_w_size - filter_w_size + 1;
    int output_h_size = input_h_size - filter_h_size + 1;
    int output_size   = output_w_size * output_h_size;
    int input_node_size  = c_size * input_h_size * input_w_size;
    int output_node_size = c_size * filter_h_size * filter_w_size;
    int output_frame_size = input_frame_size * output_size;

    std::vector<float> in_sig(input_frame_size * input_node_size);
    std::vector<float> out_sig(output_frame_size * output_node_size);

    std::mt19937_64 mt(1);
    std::normal_distribution<float>     norm_rand(0, 1);
    int idx = 0;
    for ( int frame = 0; frame < input_frame_size; ++frame ) {
        for ( int node = 0; node < input_node_size; ++node ) {
//          in_sig[node*input_frame_size + frame] = norm_rand(mt); // frame * 1000 + node;
            in_sig[node*input_frame_size + frame] = frame * 1000 + node;
        }
    }

//  for ( auto& s : in_sig ) { s = idx++; norm_rand(mt); }


    bb::NeuralNetConvolutionIm2Col<> cnvim2col(c_size, input_h_size, input_w_size, filter_h_size, filter_w_size);
    cnvim2col.SetBatchSize(input_frame_size);
    testSetupLayerBuffer(cnvim2col);
    auto in_sig_buf = cnvim2col.GetInputSignalBuffer();
    auto out_sig_buf = cnvim2col.GetOutputSignalBuffer();
    for ( int frame = 0; frame < input_frame_size; ++frame ) {
        for ( int node = 0; node < input_node_size; ++node ) {
            in_sig_buf.SetReal(frame, node, in_sig[node*input_frame_size + frame]);
        }
    }

    std::cout << "\n\n";
    std::cout << "[Im2Col]\n";
    std::cout << "batch_size    :" << input_frame_size << "\n";
    std::cout << "c_size        :" << c_size << "\n";
    std::cout << "input_h_size  :" << input_h_size << "\n";
    std::cout << "input_w_size  :" << input_w_size << "\n";
    std::cout << "filter_h_size :" << filter_h_size << "\n";
    std::cout << "filter_w_size :" << filter_w_size << "\n";
    std::cout << "total input size  : " << input_frame_size * input_node_size << "\n";
    std::cout << "total output size : " << output_frame_size * output_node_size << "\n\n";

    std::cout << "[GPU]" << std::endl;

    bbcu_eva_Im2Col_Forward(
            &in_sig[0],
            &out_sig[0],
            input_frame_size,
            input_w_size,
            input_h_size,
            c_size,
            filter_w_size,
            filter_h_size
        );

    auto fw_time0 = std::chrono::system_clock::now();
    cnvim2col.Forward();
    auto fw_time1 = std::chrono::system_clock::now();

    std::cout << "\n\n[CPU]" << std::endl;
    double fw_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(fw_time1-fw_time0).count();
    std::cout << "OpenMP : " << fw_elapsed << " [msec]" << std::endl;
    std::cout << "\n\n";


    auto o_frame_size = out_sig_buf.GetFrameSize();
    auto o_node_size  = out_sig_buf.GetNodeSize();

    for ( int frame = 0; frame < output_frame_size; ++frame ) {
        for ( int node = 0; node < output_node_size; ++node ) {
            EXPECT_EQ(out_sig_buf.GetReal(frame, node), out_sig[node*output_frame_size + frame]);
//          std::cout << out_sig_buf.GetReal(frame, node) << ", " << out_sig[node*output_frame_size + frame] << std::endl;
        }
    }



    // backward
    std::cout << "<<<backward>>\n";

    std::vector<float> in_err(input_frame_size * input_node_size);
    std::vector<float> out_err(output_frame_size * output_node_size);

    idx = 0;
    for ( int frame = 0; frame < output_frame_size; ++frame ) {
        for ( int node = 0; node < output_node_size; ++node ) {
//          out_err[node*output_frame_size + frame] = idx++; // norm_rand(mt); // frame * 1000 + node;
            out_err[node*output_frame_size + frame] = out_sig[node*output_frame_size + frame];
        }
    }

    auto in_err_buf = cnvim2col.GetInputErrorBuffer();
    auto out_err_buf = cnvim2col.GetOutputErrorBuffer();
    for ( int frame = 0; frame < output_frame_size; ++frame ) {
        for ( int node = 0; node < output_node_size; ++node ) {
            out_err_buf.SetReal(frame, node, out_err[node*output_frame_size + frame]);
        }
    }

    std::cout << "[GPU]" << std::endl;

    bbcu_eva_Im2Col_Backward(
            &in_err[0],
            &out_err[0],
            input_frame_size,
            input_w_size,
            input_h_size,
            c_size,
            filter_w_size,
            filter_h_size
        );

    auto bw_time0 = std::chrono::system_clock::now();
    cnvim2col.Backward();
    auto bw_time1 = std::chrono::system_clock::now();

    std::cout << "\n\n[CPU]" << std::endl;
    double bw_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(bw_time1-bw_time0).count();
    std::cout << "OpenMP : " << bw_elapsed << " [msec]" << std::endl;
    std::cout << "\n\n";
    
    for ( int frame = 0; frame < input_frame_size; ++frame ) {
        for ( int node = 0; node < input_node_size; ++node ) {
            EXPECT_EQ(in_err_buf.GetReal(frame, node), in_err_buf.GetReal(frame, node));
//          EXPECT_EQ(in_err_buf.GetReal(frame, node), in_err[node*input_frame_size + frame]);
//          std::cout << in_err_buf.GetReal(frame, node) << ", " << in_err[node*input_frame_size + frame] << std::endl;
        }
    }


}



#if 0
TEST(cudaIm2ColTest, testcudaIm2Col)
{
    bb::NeuralNetConvolutionIm2Col<> cnvim2col(2, 3, 4, 2, 3);
    
    cnvim2col.SetBatchSize(2);
    testSetupLayerBuffer(cnvim2col);

    auto in_sig_buf = cnvim2col.GetInputSignalBuffer();
    auto out_sig_buf = cnvim2col.GetOutputSignalBuffer();

    EXPECT_EQ(2 * 3 * 4, cnvim2col.GetInputNodeSize());
    EXPECT_EQ(2 * 2 * 3, cnvim2col.GetOutputNodeSize());
    EXPECT_EQ(2, cnvim2col.GetInputFrameSize());
    EXPECT_EQ(2 * 2 * 2, cnvim2col.GetOutputFrameSize());

    in_sig_buf.SetDimensions({ 4, 3, 2 });
    for (bb::INDEX f = 0; f < 2; ++f) {
        for (bb::INDEX c = 0; c < 2; ++c) {
            for (bb::INDEX y = 0; y < 3; ++y) {
                for (bb::INDEX x = 0; x < 4; ++x) {
                    in_sig_buf.SetReal(f, { x, y, c }, (float)(1000 * f + 100 * c + 10 * y + x));
                }
            }
        }
    }

    cnvim2col.Forward();

    out_sig_buf.SetDimensions({ 3, 2, 2 });
    EXPECT_EQ(0, out_sig_buf.GetReal(0,  { 0, 0, 0 }));
    EXPECT_EQ(1, out_sig_buf.GetReal(0,  { 1, 0, 0 }));
    EXPECT_EQ(2, out_sig_buf.GetReal(0,  { 2, 0, 0 }));
    EXPECT_EQ(10, out_sig_buf.GetReal(0, { 0, 1, 0 }));
    EXPECT_EQ(11, out_sig_buf.GetReal(0, { 1, 1, 0 }));
    EXPECT_EQ(12, out_sig_buf.GetReal(0, { 2, 1, 0 }));
    EXPECT_EQ(100, out_sig_buf.GetReal(0, { 0, 0, 1 }));
    EXPECT_EQ(101, out_sig_buf.GetReal(0, { 1, 0, 1 }));
    EXPECT_EQ(102, out_sig_buf.GetReal(0, { 2, 0, 1 }));
    EXPECT_EQ(110, out_sig_buf.GetReal(0, { 0, 1, 1 }));
    EXPECT_EQ(111, out_sig_buf.GetReal(0, { 1, 1, 1 }));
    EXPECT_EQ(112, out_sig_buf.GetReal(0, { 2, 1, 1 }));

    EXPECT_EQ(1, out_sig_buf.GetReal(1, { 0, 0, 0 }));
    EXPECT_EQ(2, out_sig_buf.GetReal(1, { 1, 0, 0 }));
    EXPECT_EQ(3, out_sig_buf.GetReal(1, { 2, 0, 0 }));
    EXPECT_EQ(11, out_sig_buf.GetReal(1, { 0, 1, 0 }));
    EXPECT_EQ(12, out_sig_buf.GetReal(1, { 1, 1, 0 }));
    EXPECT_EQ(13, out_sig_buf.GetReal(1, { 2, 1, 0 }));
    EXPECT_EQ(101, out_sig_buf.GetReal(1, { 0, 0, 1 }));
    EXPECT_EQ(102, out_sig_buf.GetReal(1, { 1, 0, 1 }));
    EXPECT_EQ(103, out_sig_buf.GetReal(1, { 2, 0, 1 }));
    EXPECT_EQ(111, out_sig_buf.GetReal(1, { 0, 1, 1 }));
    EXPECT_EQ(112, out_sig_buf.GetReal(1, { 1, 1, 1 }));
    EXPECT_EQ(113, out_sig_buf.GetReal(1, { 2, 1, 1 }));

    EXPECT_EQ(10, out_sig_buf.GetReal(2, { 0, 0, 0 }));
    EXPECT_EQ(11, out_sig_buf.GetReal(2, { 1, 0, 0 }));
    EXPECT_EQ(12, out_sig_buf.GetReal(2, { 2, 0, 0 }));
    EXPECT_EQ(20, out_sig_buf.GetReal(2, { 0, 1, 0 }));
    EXPECT_EQ(21, out_sig_buf.GetReal(2, { 1, 1, 0 }));
    EXPECT_EQ(22, out_sig_buf.GetReal(2, { 2, 1, 0 }));
    EXPECT_EQ(110, out_sig_buf.GetReal(2, { 0, 0, 1 }));
    EXPECT_EQ(111, out_sig_buf.GetReal(2, { 1, 0, 1 }));
    EXPECT_EQ(112, out_sig_buf.GetReal(2, { 2, 0, 1 }));
    EXPECT_EQ(120, out_sig_buf.GetReal(2, { 0, 1, 1 }));
    EXPECT_EQ(121, out_sig_buf.GetReal(2, { 1, 1, 1 }));
    EXPECT_EQ(122, out_sig_buf.GetReal(2, { 2, 1, 1 }));

    EXPECT_EQ(11, out_sig_buf.GetReal(3, { 0, 0, 0 }));
    EXPECT_EQ(12, out_sig_buf.GetReal(3, { 1, 0, 0 }));
    EXPECT_EQ(13, out_sig_buf.GetReal(3, { 2, 0, 0 }));
    EXPECT_EQ(21, out_sig_buf.GetReal(3, { 0, 1, 0 }));
    EXPECT_EQ(22, out_sig_buf.GetReal(3, { 1, 1, 0 }));
    EXPECT_EQ(23, out_sig_buf.GetReal(3, { 2, 1, 0 }));
    EXPECT_EQ(111, out_sig_buf.GetReal(3, { 0, 0, 1 }));
    EXPECT_EQ(112, out_sig_buf.GetReal(3, { 1, 0, 1 }));
    EXPECT_EQ(113, out_sig_buf.GetReal(3, { 2, 0, 1 }));
    EXPECT_EQ(121, out_sig_buf.GetReal(3, { 0, 1, 1 }));
    EXPECT_EQ(122, out_sig_buf.GetReal(3, { 1, 1, 1 }));
    EXPECT_EQ(123, out_sig_buf.GetReal(3, { 2, 1, 1 }));

    EXPECT_EQ(1010, out_sig_buf.GetReal(6, { 0, 0, 0 }));
    EXPECT_EQ(1011, out_sig_buf.GetReal(6, { 1, 0, 0 }));
    EXPECT_EQ(1012, out_sig_buf.GetReal(6, { 2, 0, 0 }));
    EXPECT_EQ(1020, out_sig_buf.GetReal(6, { 0, 1, 0 }));
    EXPECT_EQ(1021, out_sig_buf.GetReal(6, { 1, 1, 0 }));
    EXPECT_EQ(1022, out_sig_buf.GetReal(6, { 2, 1, 0 }));
    EXPECT_EQ(1110, out_sig_buf.GetReal(6, { 0, 0, 1 }));
    EXPECT_EQ(1111, out_sig_buf.GetReal(6, { 1, 0, 1 }));
    EXPECT_EQ(1112, out_sig_buf.GetReal(6, { 2, 0, 1 }));
    EXPECT_EQ(1120, out_sig_buf.GetReal(6, { 0, 1, 1 }));
    EXPECT_EQ(1121, out_sig_buf.GetReal(6, { 1, 1, 1 }));
    EXPECT_EQ(1122, out_sig_buf.GetReal(6, { 2, 1, 1 }));

    EXPECT_EQ(1011, out_sig_buf.GetReal(7, { 0, 0, 0 }));
    EXPECT_EQ(1012, out_sig_buf.GetReal(7, { 1, 0, 0 }));
    EXPECT_EQ(1013, out_sig_buf.GetReal(7, { 2, 0, 0 }));
    EXPECT_EQ(1021, out_sig_buf.GetReal(7, { 0, 1, 0 }));
    EXPECT_EQ(1022, out_sig_buf.GetReal(7, { 1, 1, 0 }));
    EXPECT_EQ(1023, out_sig_buf.GetReal(7, { 2, 1, 0 }));
    EXPECT_EQ(1111, out_sig_buf.GetReal(7, { 0, 0, 1 }));
    EXPECT_EQ(1112, out_sig_buf.GetReal(7, { 1, 0, 1 }));
    EXPECT_EQ(1113, out_sig_buf.GetReal(7, { 2, 0, 1 }));
    EXPECT_EQ(1121, out_sig_buf.GetReal(7, { 0, 1, 1 }));
    EXPECT_EQ(1122, out_sig_buf.GetReal(7, { 1, 1, 1 }));
    EXPECT_EQ(1123, out_sig_buf.GetReal(7, { 2, 1, 1 }));


    // backward
    auto out_err_buf = cnvim2col.GetOutputErrorBuffer();
    auto in_err_buf = cnvim2col.GetInputErrorBuffer();

    out_err_buf.SetDimensions({ 3, 2, 2 });
    for (bb::INDEX f = 0; f < 8; ++f) {
        for (bb::INDEX c = 0; c < 2; ++c) {
            for (bb::INDEX y = 0; y < 2; ++y) {
                for (bb::INDEX x = 0; x < 3; ++x) {
                    out_err_buf.SetReal(f, { x, y, c }, (float)(1000 * f + 100 * c + 10 * y + x));
                }
            }
        }
    }
    
    cnvim2col.Backward();

//  for (int i = 0; i < 2 * 3 * 4; ++i) {
//      std::cout << in_err_buf.GetReal(0, i) << std::endl;
//  }

//  EXPECT_EQ(0, in_err_buf.GetReal(0, { 0, 0, 0 }));
//  EXPECT_EQ(((0+1) + 0)+ ((1 + 2) + 1000), in_err_buf.GetReal(0, { 1, 0, 0 }));
//  EXPECT_EQ((1 + 2) + 1000, in_err_buf.GetReal(0, { 2, 0, 0 }));

}



#if 0

#include <chrono>

TEST(NeuralNetConvolutionIm2ColTest, testNeuralNetConvolutionIm2ColSpeed)
{
    // 実践的なサイズで速度比較
    bb::NeuralNetConvExpand<> cnvexp(100, 28, 28, 3, 3);
    bb::NeuralNetConvExpandM<100, 28, 28, 3, 3> cnvexpM;

    cnvexp.SetBatchSize(256);
    cnvexpM.SetBatchSize(256);
    testSetupLayerBuffer(cnvexp);
    testSetupLayerBuffer(cnvexpM);

    std::chrono::system_clock::time_point  start, end;

    start = std::chrono::system_clock::now();
    cnvexp.Forward();
    end = std::chrono::system_clock::now();

    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << elapsed << std::endl;


    start = std::chrono::system_clock::now();
    cnvexpM.Forward();
    end = std::chrono::system_clock::now();

    elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << elapsed << std::endl;
}


#endif
#endif


#endif
