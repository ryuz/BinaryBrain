============================
クイックスタート(Python)
============================

BinaryBrain は pybind11 を利用して Python からの呼び出しも可能にしています。
python3を前提としています。

pipでのインストール
------------------------

下記のコマンドでインストール可能(になる予定)です。

::

  % pip3 install binarybrain

Google Colaboratory からも

::

  !pip install binarybrain

とすることでインストール可能です。


BinaryBrainは python3 専用です。Python2 との共存環境の場合など必要に応じて pip3 を実行ください。そうでなければ pip に読み替えてください。
インストール時にソースファイルがビルドされますので、コンパイラやCUDAなどの環境は事前に整えておく必要があります。

(Windows版はバイナリwheelが提供されるかもしれません。作者環境は ver4.0.1 現在、Python 3.7.4(Windows10)、Python 3.6.9(Ubuntu 18) です)


Python用のサンプルプログラムは下記などを参照ください。

https://github.com/ryuz/BinaryBrain/tree/ver4_release/samples/python

（ipynb 形式ですので、Jupyter Notebook、Jupyter Lab、VS code、PyCharm、GoogleColab など、読める環境を準備ください。）


setup.py でのインストール
---------------------------

pip でのインストールがうまくいかない場合や、github上の最新版を試したい場合などは setup.py でのインストールも可能です。


事前準備
^^^^^^^^^^^^^^

Python版は各種データセットに PyTorch を利用しています。
事前にインストールください。

またその他必要なパッケージを事前にインストールください。pybind11 などが必須です。

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



サンプルの実行
^^^^^^^^^^^^^^^^

コマンドラインから以下のサンプルを試すことができます。

::

  % cd samples/python/mnist

  % # Simple DNN sample
  % python3 MnistDifferentiableLutSimple.py

  % # CNN sample
  % python3 MnistDifferentiableLutCnn.py

その他のサンプルは ipynb 形式で samples/python フォルダの中にあるので Jupyter Notebook などで参照ください。


Google Colaboratory での setup.py
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Google Colaboratory で利用する場合は、ランタイムのタイプを「GPU」にして、下記を実行した後にランタイムの再起動を行えば利用できるようになるはずです。

::

  !pip install pybind11
  !git clone -b ver4_release  https://github.com/ryuz/BinaryBrain.git
  %cd BinaryBrain
  !python3 setup.py install --user


