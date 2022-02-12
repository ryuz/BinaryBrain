
#include "mnist_simple.h"
#include "MnistDifferentiableLutSimpleHls.h"


ap_uint<10> mnist_simple(ap_uint<28*28> in_data)
{
#pragma HLS pipeline II=1

    auto data1 = mnist_layer1(in_data);
    auto data2 = mnist_layer2(data1);
    auto data3 = mnist_layer3(data2);
    auto data4 = mnist_layer4(data3);
    auto data5 = mnist_layer5(data4);
    auto data6 = mnist_layer6(data5);
    return data6;
}

