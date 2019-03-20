------------------------------------------------------------------------------
 BinaryBrain  MNISTサンプル
                                         Copyright (C) 2018 by Ryuji Fuchikami
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
  MnistSimpleLutMlp.cpp              LUT-Network 単純MLPサンプル
  MnistSimpleLutCnn.cpp              LUT-Network CNNサンプル
  MnistDenseAffine.cpp               通常DNNの単純MLPサンプル
  MnistSimpleMicroMlpScratch.cpp     ネットをスクラッチで書く場合のサンプル
  readme.txt                         本ファイル
  sample_mnist.sln                   Visual-C++ 2017用ソリューション
  sample_mnist.vcxproj               Visual-C++ 2017用プロジェクト
  sample_mnist.vcxproj.filters       Visual-C++ 2017用
  sample_mnist.vcxproj.user          Visual-C++ 2017用
  verilog/bb_lut.v                   LUT の Verilogモデル
  verilog/tb_mnist_lut_net.v         単純MLP LUT-Network のテストベンチ
  verilog/tb_mnist_lut_net.vtakprj   単純MLP LUT-Network のVeritakプロジェクト


【ビルド方法】
 [Linuxの場合]
  make all

  でビルドすると 実行ファイル sample_mnist が出来ます

  なお、ここで
  make WITH_CUDA=No all
  とすると、CUDA無しのCPU版がビルドされます


  単純に実行するには

  make WITH_CUDA=Yes run

  とすると、MNIST データがなければダウンロードしてから
  ./sample_mnist All

  が実行されます。

  sample_mnist の引数は

  LutMlp                   LUT-Networkの単純多層パーセプトロンを実行
  LutCnn                   LUT-NetworkのCNNを実行
  DenseAffine              普通の単純多層パーセプトロンを実行
  SimpleMicroMlpScratch    単純多層パーセプトロンをベタ書したサンプルを実行
  All                      上のすべてを実行

  となっています。


 [Windowsの場合]
  Visual C++ 2017 でビルドできます。
  MNISTファイルなどは手動ダウンロードが必要です。



【MLP の Verilog シミュレーションまで】

  ./sample_mnist LutMlp

  を実行すると、学習完了後 verilog ディレクトリに

  mnist_train.txt       トレーニングデータ
  mnist_test.txt        評価データ
  MnistSimpleLutMlp.v   学習済みの RTL

  が出力されます。

  下記を、何らかのシミュレータでシミュレーション実行すると、
学習結果が試せます。

  tb_mnist_lut_net.v
  MnistSimpleLutMlp.v
  bb_lut.v

  tb_mnist_lut_net.vtakprj が Veritak 用のプロジェクトとなっておりますので、
Veritak ご利用のユーザーはすぐに試すことが出来ます。



【CNN の Verilog シミュレーションまで】
  現在まだVer3に移植が終わっていません。
  しばらくお待ちください m(_ _)m



------------------------------------------------------------------------------
 end of file
------------------------------------------------------------------------------
