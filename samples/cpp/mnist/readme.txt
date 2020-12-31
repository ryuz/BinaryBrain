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
  Makefile                              Linux用メイクファイル
  get_nmist.bat                         NMISTダウンロード用(Windows+cygwinなど)
  get_nmist.sh                          NMISTダウンロード用(Linux)
  main.cpp                              main関数
  MnistStochasticLutSimple.cpp          確率的LUT方式 Binary LUT-Network Simple DNNサンプル
  MnistStochasticLutCnn.cpp             確率的LUT方式 Binary LUT-Network CNNサンプル
  MnistSparseLutSimple.cpp              疎結合LUT方式 Binary LUT-Network Simple DNNサンプル
  MnistSparseLutCnn.cpp                 疎結合LUT方式 Binary LUT-Network CNNサンプル
  MnistMicroMlpLutSimple.cpp            uMLP方式 Binary LUT-Network Simple DNNサンプル
  MnistMicroMlpLutCnn.cpp               uMLP方式 Binary LUT-Network CNNサンプル
  MnistDenseSimple.cpp                  FP32の全結合DNNの MLPサンプル
  MnistDenseCnn.cpp                     FP32の全結合CNNの CNNサンプル
  MnistAeSparseLutSimple.cpp            AutoEncoder LUT-Network Simple DNNサンプル
  MnistAeSparseLutCnn.cpp               AutoEncoder LUT-Network CNNサンプル
  MnistCustomModel.cpp                  カスタムモデル作成用サンプル
  readme.txt                            本ファイル
  sample_mnist.sln                      Visual-C++ 2019用ソリューション
  sample_mnist.vcxproj                  Visual-C++ 2019用プロジェクト
  sample_mnist.vcxproj.filters          Visual-C++ 2019用
  sample_mnist.vcxproj.user             Visual-C++ 2019用
  verilog/bb_lut.v                      LUT の Verilogモデル
  verilog/tb_mnist_lut_simple.v         Simple DNN LUT-Network のテストベンチ
  verilog/tb_mnist_lut_simple.vtakprj   Simple DNN LUT-Network のVeritakプロジェクト
  verilog/iverilog_lut_simple.bat       Simple DNN LUT-Network のiverilog実行(Win)
  verilog/iverilog_lut_simple.sh        Simple DNN LUT-Network のiverilog実行(Linux)
  verilog/iverilog_lut_simple_cmd.txt   Simple DNN LUT-Network のiverilogコマンド
  verilog/tb_mnist_lut_cnn.v            CNN LUT-Network のテストベンチ
  verilog/tb_mnist_lut_cnn.vtakprj      CNN LUT-Network のVeritakプロジェクト
  verilog/iverilog_lut_cnn.bat          CNN LUT-Network のiverilog実行(Win)
  verilog/iverilog_lut_cnn.sh           CNN LUT-Network のiverilog実行(Linux)
  verilog/iverilog_lut_cnn_cmd.txt      CNN LUT-Network のiverilogコマンド
  verilog/video_mnist_cnn.v             CNNモジュール
  verilog/video_mnist_cnn_core.v        CNNモジュールのコア
  verilog/video_dnn_max_count.v         クラスタリング結果のカウンティング
  verilog/video_mnist_color.v           結果着色モジュール
  verilog/video_mnist_color_core.v      結果着色モジュールのコア


【ビルド方法】
 [Linuxの場合]
  make all

  でビルドすると 実行ファイル sample-mnist が出来ます

  なお、ここで
  make WITH_CUDA=No all
  とすると、CUDA無しのCPU版がビルドされます

  make dl_data

  と実行すると、MNISTのデータをダウンロードします。

  試し実行するには

  ./sample-mnist All

  とすると、すべての内蔵サンプルが順番に実行されます。

  sample_mnist の引数は

  StochasticLutSimple      確率的LUT-Networkの単純DNNを実行
  StochasticLutCnn         確率的LUT-NetworkのCNNを実行
  SparseLutSimple          疎結合LUT-Networkの単純DNNを実行
  SparseLutCnn             疎結合LUT-NetworkのCNNを実行
  MicroMlpLutSimple        μMLP方式のLUT-Networkの単純DNNを実行
  MicroMlpLutCnn           μMLP方式のLUT-NetworkのCNNを実行
  DenseMlp                 全結合の単純DNNを実行
  DenseCnn                 全結合のCNNを実行
  AeSparseLutSimple        AutoEncoderを疎結合LUT-Networkの単純DNNで実行
  AeSparseLutCnn           AutoEncoderを疎結合LUT-NetworkのCNNで実行
  All                      上のすべてを実行

  となっており、試したいモデルだけ実行することも可能です。
  また -epoch オプションなどで epoch 数の指定も可能です。詳しくは main.cpp を確認ください。


 [Windowsの場合]
  Visual C++ 2019 でビルドできます。
  MNISTファイルなどは手動ダウンロードが必要です。



【Simple DNN の Verilog シミュレーションまで】

  ./sample-mnist SparseLutSimple

  を実行すると、学習完了後 verilog ディレクトリの下に

  mnist_train.txt            トレーニングデータ
  mnist_test.txt             評価データ
  MnistSparseLutSimple.v     学習済みの RTL

  が出力されます。

  下記を、何らかのシミュレータでシミュレーション実行すると、
学習結果が試せます。

  tb_mnist_lut_simple.v
  MnistSparseLutSimple.v
  bb_lut.v

  Vivadoシミュレータ(xsim)を利用する場合は、xsim_lut_simple.bat が利用可能です。

  tb_mnist_lut_simple.vtakprj が Veritak 用のプロジェクトとなっておりますので、
Windowsで Veritak ご利用のユーザーは活用ください。

  iverilog(Icarus Verilog)用に iverilog_lut_simple.sh というスクリプトも
用意しています(が、ネットワークの特性か結構遅いです)。


【CNN の Verilog シミュレーションまで】

  ./sample-mnist SparseLutCnn

  を実行すると、学習完了後 verilog ディレクトリの下に

  mnist_test_160x120.ppm  テスト画像(160x120)
  mnist_test_640x480.ppm  テスト画像(640x480)
  MnistSparseLutCnn.v     学習済みの RTL

  tb_mnist_lut_cnn.vtakprj が Veritak 用のプロジェクトとなっておりますので、
Windowsで Veritak ご利用のユーザーは活用ください。

  iverilog(Icarus Verilog)用に iverilog_lut_cnn.sh というスクリプトも
用意しています(が、ネットワークの特性か結構遅いです)。

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
