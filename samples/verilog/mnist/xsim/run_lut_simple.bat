call xelab -prj project_lut_simple.prj -debug wave tb_mnist_lut_simple -s tb_mnist_lut_simple
copy ..\mnist_test.txt .
call xsim tb_mnist_lut_simple -t xsim_run_all.tcl
