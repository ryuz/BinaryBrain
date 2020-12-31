iverilog -o tb_mnist_lut_simple.vvp -s tb_mnist_lut_simple -c cmd_lut_simple.txt
copy ..\mnist_test.txt .
vvp tb_mnist_lut_simple.vvp
