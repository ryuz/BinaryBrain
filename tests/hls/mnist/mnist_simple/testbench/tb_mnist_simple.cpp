
#include <iostream>
#include <fstream>
#include "mnist_simple.h"

int main()
{
    // データファイル読み込み
    std::ifstream is("../../../../testbench/mnist_hls_test.txt");
    if( !is ) { std::cout << "open error : mnist_hls_test.txt" << std::endl; return 1; }
    
    int n = 0;
    int ok = 0;
    for ( int i = 0; i < 64; ++i ) {
        // データ読み込み
        ap_uint<28*28> in_data;
        int label;
        is >> label;
        for ( int j = 0; j < 28*28; ++j ) {
            int val;
            is >> val;
            in_data[j] = val;
        }

        // テスト
        auto out_data = mnist_simple(in_data);

        // 確認
        ap_uint<10>    exp_data = (1 << label);
        if ( out_data == exp_data ) {
            std::cout << "[OK]   ";
            ok++;
        } else {
            std::cout << "[miss] ";
        }
        std::cout << "label: " << std::dec << label << " out: 0x" << std::hex << (int)out_data << std::endl;
        n++;
    }
    std::cout << "total : " << std::dec << ok << "/" << n << std::endl;

    // とりあえず1/3整合していればOKとする
    assert(ok >= n/3);

    return 0;
}

