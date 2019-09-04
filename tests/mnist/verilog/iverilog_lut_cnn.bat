#!/bin/sh
iverilog -o tb_mnist_lut_cnn.vvp -s tb_mnist_lut_cnn -c iverilog_lut_cnn_cmd.txt
vvp tb_mnist_lut_cnn.vvp
