#! /bin/bash -eu

rm -fr xsim.dir
rm -fr .Xil
rm -f webtalk*.jou
rm -f webtalk*.log
rm -f xvlog*.log
rm -f xvlog*.pb
rm -f xelab*.log
rm -f xelab*.pb
rm -f xsim*.jou
rm -f xsim*.log

TOP_MODULE=tb_mnist_lut4_simple

xvlog -f xvlog_cmd.txt
xelab -debug wave $TOP_MODULE -s $TOP_MODULE
xsim $TOP_MODULE -t xsim_run_all.tcl
