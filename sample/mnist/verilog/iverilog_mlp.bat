iverilog -o tb_mnist_lut_mlp.vvp -s tb_mnist_lut_net -c iverilog_mlp_cmd.txt
vvp tb_mnist_lut_mlp.vvp
