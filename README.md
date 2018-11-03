# BinaryBrain for LUT(Look-up table) Networks platform

"LUT Networks" is one of the "Binary Deep Neural Networks" for FPGA.

http://ryuz.txt-nifty.com/blog/2018/10/binary-deep-neu.html


## sample program (MNIST)
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
% cd make all
% cd make run
```


## What is LUT networks?
FPGAのLUTをダイレクトに学習させることを目標とした、バイナリニューラルネット用のライブラリを整理中です。
基本的にはヘッダファイルのみのライブラリとする見込みです。
AXV2以降の命令が使えるCPUと、Windows7以降の環境を想定しております。

CEREAL や EIGEN など、submoduleとして、他のライブラリを用いていますので、それらは個別にライセンスを確認ください。
現在 git にあるオリジナルのソースコード部分のライセンスは MIT ライセンスとしております。



