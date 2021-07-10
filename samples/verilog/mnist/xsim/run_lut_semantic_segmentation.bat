call xelab -prj project_lut_semantic_segmentation.prj -debug wave tb_mnist_lut_semantic_segmentation -s tb_mnist_lut_semantic_segmentation
copy ..\mnist_test_160x120.ppm .
copy ..\mnist_test_640x480.ppm .
call xsim tb_mnist_lut_semantic_segmentation -t xsim_run_all.tcl
