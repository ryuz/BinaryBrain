#!/bin/sh
xelab -prj xsim_lut_mlp.prj -debug wave tb_mnist_lut_mlp -s tb_mnist_lut_mlp
xsim tb_mnist_lut_mlp -t xsim_lut_mlp.tcl
