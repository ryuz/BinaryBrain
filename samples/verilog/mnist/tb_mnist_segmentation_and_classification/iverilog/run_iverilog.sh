#! /bin/bash -eu

TOP_MODULE=tb_mnist_lut_segmentation_and_classification

iverilog -o $TOP_MODULE.vvp -s $TOP_MODULE -c iverilog_cmd.txt -DIVERILOG
vvp $TOP_MODULE.vvp
