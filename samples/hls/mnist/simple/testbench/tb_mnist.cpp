
#include "mnist_sample.h"

int main()
{
    std::cout << "start testbench" << std::endl;

    ap_uint<1>  in[28*28];
    ap_uint<4>  out[0];
    mnist_sample(in, out);
    
    return 0;
}
