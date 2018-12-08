# BinaryBrain <br> --binary neural networks evaluation platform for LUT-networks

## High Performance Binary Neural Networks for FPGA
"LUT(Look-up Table) Networks" is one of the "Binary Deep Neural Networks" for FPGA.
The FPGA's LUT can learn direct from LUT-network on this platform.

## How to use sample program (MNIST)
### windows
1. install VisualStudio 2015. 
2. git clone --recursive https://github.com/ryuz/BinaryBrain.git 
3. download MNIST from http://yann.lecun.com/exdb/mnist/
4. decompress MNIST for "\sample\mnist"
5. open VC++ solution "sample\mnist\sample_mnist.sln"
6. build "x64 Release"
7. run

### Linux(Ubuntu 18.04.1)
1. install tools 
```
% sudo apt install make
% sudo apt install g++
% sudo apt install clang
% sudo apt install git
```
2. build and run
```
% git clone --recursive https://github.com/ryuz/BinaryBrain.git
% cd BinaryBrain/sample/mnist
% make all
% make run
```


## What is LUT networks?
### Design flow
FPGA circuit is constructed from LUTs.
This platform let the LUT's table learn directly.

![LutNet_design_flow.png](documents/images/LutNet_design_flow.png "design flow")

### layer model
A single LUT can calcurate XOR. But a single perceptron node can't learn XOR.
LUT-Network's one node has a hidden perceptrons.

![LutNet_layer_model.png](documents/images/LutNet_layer_model.png "layer_model")

### LUT node model
One LUT's equivalent model is many perceptron that has hidden layer.

![LutNet_lut_equivalent_model.png](documents/images/LutNet_lut_equivalent_model.png "LUT node model")

Learning model of LUT-Network is shown below.
![LutNet_lut_node_model.png](documents/images/LutNet_node_model.png "LUT node model")

### Difference from other binary deep neural network
Though LUT-Network is binary nwtwork, it has FP32 weight parameter. Only activation layer's output is binary.


![difference_other_networks.png](documents/images/difference_other_networks.png "difference from other networks")

### Performance  estimations
LUT-network's learning cost is heavy, but prediction computing performanse of FPGA is very high.

![performance.png](documents/images/performance.png "parformance")

## License
This source code's license is MIT license.

(Note : This program using Eigen and CEREAL)

## Reference
- BinaryConnect: Training Deep Neural Networks with binary weights during propagations<br>
https://arxiv.org/pdf/1511.00363.pdf

- Binarized Neural Networks<br>
https://arxiv.org/abs/1602.02505

- Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1<br>
https://arxiv.org/abs/1602.02830

- XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks<br>
https://arxiv.org/abs/1603.05279

- Xilinx UltraScale Architecture Configurable Logic Block User Guide<br>
https://japan.xilinx.com/support/documentation/user_guides/ug574-ultrascale-clb.pdf


## Author's information
Ryuji Fuchikami
- github : https://github.com/ryuz
- blog : http://ryuz.txt-nifty.com
- twitter : https://twitter.com/ryuz88
- facebook : https://www.facebook.com/ryuji.fuchikami
- web-site : http://ryuz.my.coocan.jp/
- e-mail : ryuji.fuchikami@nifty.com


