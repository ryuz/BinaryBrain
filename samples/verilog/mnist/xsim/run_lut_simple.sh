#!/bin/sh
xelab -prj xsim_lut_simple.prj -debug wave tb_mnist_lut_simple -s tb_mnist_lut_simple
xsim tb_mnist_lut_simple -t xsim_run_all.tcl
