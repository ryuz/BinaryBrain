==============
はじめに
==============


概要
=======

BinaryBrain は主に当サイトが研究中の LUT(Look-Up Table)-Networkを実験することを目的に作成した
ディープラーニング用のプラットフォームです。

LUT-Networkの評価を目的に作成しておりますが、それ以外の用途にも利用可能です。

以下の特徴があります

- ニューラルネットのFPGA化をメインターゲットにしている
- バイナリネットであるも関わらず変調技術によりAutoencodeや回帰分析が可能
- 独自のSparse-LUTモデルにより、LUTの性能を最大限引き出したが学習できる
- 量子化＆疎行列のネットワークでパフォーマンスの良い学習が出来る環境を目指している
- C++で記述されている
- GPU(CUDA)に対応している
- 高速でマニアックな自作レイヤーが作りやすい
- Pythonからの利用も可能


クイックスタート(C++)
=====================

まずはじめに付属のMNISTサンプルを動かすまでを紹介します。

AXV2以降の命令が使えるCPUと、Windows7以降もしくは Linuxの環境を想定しております。
CUDAにも対応していまが、nvccが利用可能な環境でビルドする必要があります。

CUDAについてはNVIDIAのページを参考に事前にインストールください。
https://developer.nvidia.com/cuda-downloads

なお make 時に make WITH_CUDA=No と指定することで、GPUを使わないCPU版もビルド可能です。


Windows
-----------
1. install VisualStudio 2017 + CUDA 10.1
2. git clone --recursive -b ver3_release https://github.com/ryuz/BinaryBrain.git
3. download MNIST from http://yann.lecun.com/exdb/mnist/
4. decompress MNIST for "\samples\mnist"
5. open VC++ solution "samples\mnist\sample_mnist.sln"
6. build "x64 Release"
7. run

Linux(Ubuntu 18.04.1)
----------------------

install tools
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

  % git clone --recursive -b ver3_release  https://github.com/ryuz/BinaryBrain.git
  % cd BinaryBrain/samples/mnist
  % make
  % make dl_data
  % ./sample-mnist All


Google Colaboratory
---------------------------

nvcc が利用可能な Google Colaboratory でも動作可能なようです。
以下あくまで参考ですが、ランタイムのタイプをGPUに設定した上で、下記のような操作で、ビルドして動作させることができます。

::

  !git clone --recursive -b ver3_release  https://github.com/ryuz/BinaryBrain.git
  %cd BinaryBrain/samples/mnist
  !make all
  !make run



クイックスタート(Python)
============================

BinaryBrain は pybind11 を利用して Python からの呼び出しも可能にしています。
python3を前提としています。

pipでのインストール
------------------------

下記のコマンドでインストール可能です。

::

  % pip3 install binarybrain

Windowsなど環境によっては pip3 が存在せず、pip のみ場合は pip3 を pip に置き換えて実行ください。
インストール時にソースファイルがビルドされますので、コンパイラやCUDAなどの環境は事前に整えておく必要があります。
(Windows版はバイナリwheelが提供されるかもしれません)

Python用のサンプルプログラムは下記などを参照ください。

https://github.com/ryuz/BinaryBrain/tree/master/python/samples




setup.py でのインストール
---------------------------

事前準備
^^^^^^^^^^^^^^
必要なパッケージを事前にインストールください

::

  % pip3 install setuptools
  % pip3 install pybind11
  % pip3 install numpy
  % pip3 install tqdm


Windows環境の場合、nvccのほかにも VisualStudio の 64bit 版がコマンドラインから利用できるようにしておく必要があります。
例えば以下のように実行しておきます。 x64 の指定が重要です。

::

  > "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

インストール
^^^^^^^^^^^^^^

下記のコマンドでインストール可能です。

::

  % # install
  % cd python
  % python3 setup.py install



githubについて
============================

現在 version3 は下記の branch で管理しています

ver3_develop
  開発用ブランチです。ビルド不能な状態になることもあります。
  最新のコードにアクセスしたい場合はここをご覧ください。

ver3_release
  リリース作成用ブランチです。

master
  リリースブランチで確認したものを反映。

tag は 開発都合で ver3_build0001 のような形式で定期的に打っており、
リリースのタイミングでバージョン番号のタグを打つようにしております。
(以前はリリースごとにver3_release1 のような形で打つように
していました)。

まだ、開発初期で仕様が安定していませんので、再現性の確保などが
必要な際はタグを活用ください。


基本的な使い方
=================

基本的には C++ や Python で、ネットワークを記述し、学習を行った後に
その結果を verilog などで出力して、FPGA化することを目的に作成しています。

もちろんBinaryBrain自体は学習によってネットワークのパラメータも求めるまでが
主体ですので、その結果を使ってC言語を出力するルーチンをユーザー側で開発することも
自由です。

C++用のCPU版に関してはヘッダオンリーライブラリとなっているため、include 以下にある
ヘッダファイルをインクルードするだけでご利用いただけます。
GPUを使う場合は、ヘッダ読み込みの際に BB_WITH_CUDA マクロを定義した上で、cuda 以下にある
ライブラリをビルドした上でリンクする必要があります。

また、BB_WITH_CEREAL マクロを定義すると、途中経過の保存形式に json が利用可能となります。

Python版を使う場合は、import するだけで利用可能です。

使い方はsampleなどを参考にしてください。

