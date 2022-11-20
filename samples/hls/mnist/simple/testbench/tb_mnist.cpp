
#include "mnist_sample.h"
#include "mnist_test_data.h"

int main()
{
    std::cout << "start testbench" << std::endl;

    int n = 0;
    int ok = 0;
    for ( int i = 0; i < 20; ++i ) {
        ap_uint<1>  in[28*28];
        for ( int y = 0; y < 28; ++y ) {
            for ( int x = 0; x < 28; ++x ) {
                in[y*28+x] = test_images[i][y][x];
            }
        }

        ap_uint<4>  out[0];
        mnist_sample(in, out);

        n++;
        if ( out[0] == test_labels[i] ) {
            ok++;
        }

        std::cout << "out[" << i << "]=" << (int)out[0] << " exp:"<< (int)test_labels[i] << "  " << (out[0] == test_labels[i] ? "ok" : "miss") << std::endl; 
    }
    std::cout << "accuracy = " << ok << "/" << n << std::endl; 
    
    return 0;
}
