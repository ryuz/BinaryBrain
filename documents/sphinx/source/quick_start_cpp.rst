==============================
クイックスタート(C++)
==============================


まずはじめに付属のMNISTサンプルを動かすまでを紹介します。

AXV2以降の命令が使えるCPUと、Windows7以降もしくは Linuxの環境を想定しております。
CUDAにも対応していまが、nvccが利用可能な環境でビルドする必要があります。

CUDAについてはNVIDIAのページを参考に事前にインストールください。
https://developer.nvidia.com/cuda-downloads

なお make 時に make WITH_CUDA=No と指定することで、GPUを使わないCPU版もビルド可能です。


Windows
-----------
1. install VisualStudio 2019 + CUDA 10.1
2. git clone --recursive -b ver4_release https://github.com/ryuz/BinaryBrain.git
3. download MNIST from http://yann.lecun.com/exdb/mnist/
4. decompress MNIST for "\samples\cpp\mnist"
5. open VC++ solution "samples\cpp\mnist\sample_mnist.sln"
6. build "x64 Release"
7. run

Linux(Ubuntu 18.04.1)
----------------------

1. install tools
^^^^^^^^^^^^^^^^^

::

  % sudo apt update
  % sudo apt upgrade
  % sudo apt install git
  % sudo apt install make
  % sudo apt install g++
  % # sudo apt install nvidia-cuda-toolkit
  % wget http://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.243_418.87.00_linux.run
  % sudo sh cuda_10.1.243_418.87.00_linux.run

2. build and run
^^^^^^^^^^^^^^^^^

::

  % git clone --recursive -b ver4_release  https://github.com/ryuz/BinaryBrain.git
  % cd BinaryBrain/samples/cpp/mnist
  % make
  % make dl_data
  % ./sample-mnist All


ここで単に

::

  % ./sample-mnist

と打ち込むと、使い方が表示されます。


Google Colaboratory
---------------------------

nvcc が利用可能な Google Colaboratory でも動作可能なようです。
以下あくまで参考ですが、ランタイムのタイプをGPUに設定した上で、下記のような操作で、ビルドして動作させることができます。

::

  !git clone --recursive -b ver4_release  https://github.com/ryuz/BinaryBrain.git
  %cd BinaryBrain/samples/cpp/mnist
  !make all
  !make run

