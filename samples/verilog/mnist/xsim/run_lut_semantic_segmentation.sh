#!/bin/sh
xelab -prj xsim_lut_semantic_segmentation.prj -debug wave tb_mnist_lut_semantic_segmentation -s tb_mnist_lut_semantic_segmentation
xsim tb_mnist_lut_semantic_segmentation -t xsim_run_all.tcl
