#!/bin/sh
iverilog -o tb_mnist_lut_mlp.vvp -s tb_mnist_lut_mlp -c cmd_lut_mlp.txt
cp ../mnist_test.txt .
vvp tb_mnist_lut_mlp.vvp
