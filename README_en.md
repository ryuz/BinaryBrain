
# BinaryBrain Version 3<br> --binary neural networks platform for LUT-networks

[Japanese version](README.md)

## Overview
BinaryBrain is a platform for deep learning. It can train to LUT(Look-up Table)-Network.
LUT-Network is one of binary neural networks.

It has the following features

- The main target is FPGA(field-programmable gate array).
- Regression analysis is possible though it is binary model.
- Efficient learning with original Stocastic-LUT model.
- It can compute sparse matrix with high performance.
- Developed in C++
- Accelerated with GPU (CUDA)


## How to use sample program (MNIST)
### windows
1. install VisualStudio 2017 + CUDA 10.1
2. git clone --recursive -b ver3_release https://github.com/ryuz/BinaryBrain.git 
3. download MNIST from http://yann.lecun.com/exdb/mnist/
4. decompress MNIST for "\sample\mnist"
5. open VC++ solution "sample\mnist\sample_mnist.sln"
6. build "x64 Release"
7. run

### Linux(Ubuntu 18.04.1)
1. install tools 
```
% sudo apt update
% sudo apt upgrade
% sudo apt install git
% sudo apt install make
% sudo apt install g++
% sudo apt install nvidia-cuda-toolkit
```
2. build and run
```
% git clone --recursive -b ver3_release  https://github.com/ryuz/BinaryBrain.git
% cd BinaryBrain/sample/mnist
% make
% make dl_data
% ./sample-mnist All
```

If you don't use GPU, please add "WITH_CUDA=No" option to make.

### Google Colaboratory
Currently you can use nvcc on Google Colaboratory.
Please select GPU runtime.
```
!git clone --recursive -b ver3_release  https://github.com/ryuz/BinaryBrain.git
%cd BinaryBrain/sample/mnist
!make all
!make run
```
You can build C++ source code from iPython Notebook.


## What is LUT networks?

There is also a document on [slideshare](https://www.slideshare.net/ryuz88/lutnetwork-revision2-english-version).

### Design flow
FPGA circuit is constructed from LUTs.
This platform let the LUT's table learn directly.

![LutNet_design_flow.png](documents/images/LutNet_design_flow.png "design flow")

### Difference from other binary deep neural network
Though LUT-Network is binary nwtwork, it has FP32 weight parameter.

![difference_other_networks.png](documents/images/difference_other_networks.png "difference from other networks")

high-performance prediction, and fast learning.

### Binary modulation model
BinaryBrain can handle binary modulated models.
The binary modulation model is as follows.

![modulation_model.png](documents/images/modulation_model.png "modulation_model")

For example, PWM(Pulse Width Modulation), delta sigma modulation, and Digital amplifier are also a kind of binary modulation.

### Stochastic-LUT model
I invented Stochastic-LUT model to train LUT-Network.
The Stochastic-LUT inputs binary stochastic variables and outputs binary stochastic variables.
A stochastic variables are expressed in FP32.

The model of 2 input LUT (table sizes is 4) is shown below as an example.

![stochastic_lut2.png](documents/images/stochastic_lut2.png "stochastic_lut2")

x0-x1 is input stochastic variables. W0-W3 is table value.

- Probability that W0 is selected : (1 - x1) * (1 - x0)
- Probability that W1 is selected : (1 - x1) * x0
- Probability that W2 is selected : x1 * (1 - x0)
- Probability that W3 is selected : x1 * x0

 and, output stochastic variables y is shown below.

y =   W0 * (1 - x1) * (1 - x0)
      + W1 * (1 - x1) * x0
      + W2 * x1 * (1 - x0)
      + W3 * x1 * x0

 Because this calculation tree is differentiable, it can be calculate back-propagation.
The formula for the 6-input LUT is larger, but can be calculated in the same method.

By using the Stochastic-LUT model, it is possible to perform learning much faster and with higher accuracy than the micro-MLP model described later.


### micro-MLP model
The micro-MLP model is a learning model of LUT using conventional perceptron.
A single LUT can calcurate XOR. But a single perceptron node can't learn XOR.
LUT-Network's one node has a hidden perceptrons.

![LutNet_layer_model.png](documents/images/LutNet_layer_model.png "layer_model")

### LUT node model
One LUT's equivalent model is many perceptron that has hidden layer.

![LutNet_lut_equivalent_model.png](documents/images/LutNet_lut_equivalent_model.png "LUT node model")

Learning model of LUT-Network is shown below.
![LutNet_lut_node_model.png](documents/images/LutNet_node_model.png "LUT node model")


### Performance  estimations

![performance.png](documents/images/performance.png "parformance")

This estimate is when using a Stochastic-LUT.

## License
This source code's license is MIT license.

(Note : This program using CEREAL)

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


