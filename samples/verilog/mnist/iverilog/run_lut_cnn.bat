iverilog -o tb_mnist_lut_cnn.vvp -s tb_mnist_lut_cnn -c cmd_lut_cnn.txt

cp ..\mnist_test_160x120.ppm .
cp ..\mnist_test_640x480.ppm .

vvp tb_mnist_lut_cnn.vvp
