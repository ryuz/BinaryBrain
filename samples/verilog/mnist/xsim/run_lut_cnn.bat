call xelab -prj project_lut_cnn.prj -debug wave tb_mnist_lut_cnn -s tb_mnist_lut_cnn
copy ..\*.ppm .
call xsim tb_mnist_lut_cnn -t xsim_run_all.tcl
