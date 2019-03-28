------------------------------------------------------------------------------
 BinaryBrain  MNISTサンプル
                                    Copyright (C) 2018-2019 by Ryuji Fuchikami
                                    https://github.com/ryuz
                                    ryuji.fuchikami@nifty.com
------------------------------------------------------------------------------


【概要】
  本ディレクトリは LUT-Network にて MNIST データを学習するサンプル一式と
なります。
  動かし方を説明するもので、学習時間を短めに設定しており、認識率などは
高くありませんのであらかじめご了承ください。


【ファイル構成】
  Makefile                           Linux用メイクファイル
  get_nmist.bat                      NMISTダウンロード用(Windows+cygwinなど)
  get_nmist.sh                       NMISTダウンロード用(Linux)
  main.cpp                           main関数
  MnistLutMlp.cpp                    Binary LUT-Network MLPサンプル
  MnistLutCnn.cpp                    Binary LUT-Network CNNサンプル
  MnistDenseMlp.cpp                  FP32の全結合DNNの MLPサンプル
  MnistDenseCnn.cpp                  FP32の全結合CNNの CNNサンプル
  readme.txt                         本ファイル
  sample_mnist.sln                   Visual-C++ 2017用ソリューション
  sample_mnist.vcxproj               Visual-C++ 2017用プロジェクト
  sample_mnist.vcxproj.filters       Visual-C++ 2017用
  sample_mnist.vcxproj.user          Visual-C++ 2017用
  verilog/bb_lut.v                   LUT の Verilogモデル
  verilog/tb_mnist_lut_mlp.v         MLP LUT-Network のテストベンチ
  verilog/tb_mnist_lut_mlp.vtakprj   MLP LUT-Network のVeritakプロジェクト
  verilog/iverilog_lut_mlp.bat       MLP LUT-Network のiverilog実行(Win)
  verilog/iverilog_lut_mlp.sh        MLP LUT-Network のiverilog実行(Linux)
  verilog/iverilog_lut_mlp_cmd.txt   MLP LUT-Network のiverilogコマンド
  verilog/tb_mnist_lut_cnn.v         CNN LUT-Network のテストベンチ
  verilog/tb_mnist_lut_cnn.vtakprj   CNN LUT-Network のVeritakプロジェクト
  verilog/iverilog_lut_cnn.bat       CNN LUT-Network のiverilog実行(Win)
  verilog/iverilog_lut_cnn.sh        CNN LUT-Network のiverilog実行(Linux)
  verilog/iverilog_lut_cnn_cmd.txt   CNN LUT-Network のiverilogコマンド
  verilog/video_mnist_cnn.v          CNNモジュール
  verilog/video_mnist_cnn_core.v     CNNモジュールのコア
  verilog/video_dnn_max_count.v      クラスタリング結果のカウンティング
  verilog/video_mnist_color.v        結果着色モジュール
  verilog/video_mnist_color_core.v   結果着色モジュールのコア


【ビルド方法】
 [Linuxの場合]
  make all

  でビルドすると 実行ファイル sample-mnist が出来ます

  なお、ここで
  make WITH_CUDA=No all
  とすると、CUDA無しのCPU版がビルドされます

  make dl_mnist

  と実行すると、MNISTのデータをダウンロードします。

  試し実行するには

  ./sample-mnist All

  とすると、すべての内蔵サンプルが順番に実行されます。

  sample_mnist の引数は

  LutMlp                   LUT-Networkの多層パーセプトロンを実行
  LutCnn                   LUT-NetworkのCNNを実行
  DenseMlp                 FP32全結合の多層パーセプトロンを実行
  DenseCnn                 FP32全結合のCNNを実行
  All                      上のすべてを実行

  となっており、試したいモデルだけ実行することも可能です。
  また -epoch オプションなどで epoch 数の指定も可能です。詳しくは main.cpp を確認ください。


 [Windowsの場合]
  Visual C++ 2017 でビルドできます。
  MNISTファイルなどは手動ダウンロードが必要です。



【MLP の Verilog シミュレーションまで】

  ./sample-mnist LutMlp

  を実行すると、学習完了後 verilog ディレクトリの下に

  mnist_train.txt  トレーニングデータ
  mnist_test.txt   評価データ
  MnistLutMlp.v    学習済みの RTL

  が出力されます。

  下記を、何らかのシミュレータでシミュレーション実行すると、
学習結果が試せます。

  tb_mnist_lut_mlp.v
  bb_lut.v
  MnistLutMlp.v

  iverilog(Icarus Verilog)用に iverilog_lut_mlp.sh というスクリプトも
用意しています(が、ネットワークの特性か結構遅いです)。

  tb_mnist_lut_mlp.vtakprj が Veritak 用のプロジェクトとなっておりますので、
Windowsで Veritak ご利用のユーザーは活用ください。



【CNN の Verilog シミュレーションまで】

  ./sample-mnist LutCnn

  を実行すると、学習完了後 verilog ディレクトリの下に

  mnist_test_160x120.ppm  テスト画像(160x120)
  mnist_test_640x480.ppm  テスト画像(640x480)
  MnistLutCnn.v           学習済みの RTL

  iverilog(Icarus Verilog)用に iverilog_lut_cnn.sh というスクリプトも
用意しています(が、ネットワークの特性か結構遅いです)。

  tb_mnist_lut_cnn.vtakprj が Veritak 用のプロジェクトとなっておりますので、
Windowsで Veritak ご利用のユーザーは活用ください。

  CNN の動作には Jelly(https://github.com/ryuz/jelly) を利用しており、
git の submodule にて取り込んで、必要なファイルを参照していますので
ご注意ください。

  かなり時間が掛かりますが、うまく行けば col_0.ppm などが、ppm形式の色づけされた
結果画像として出力されます。
  なお MaxPooling を 2回通しているため、出力サイズは縦横それぞれ元の
サイズの 1/4 となります。

  色は  0:黒 1:茶 2:赤 3:橙 4:黄 5:緑 6:青 7:紫 8:灰 9:白 のカラーコードで
着色しています。


------------------------------------------------------------------------------
 end of file
------------------------------------------------------------------------------
