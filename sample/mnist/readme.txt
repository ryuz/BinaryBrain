MNISTサンプル

【基本】
[Linuxの場合]
  make WITH_CUDA=Yes all

  でビルドすると 実行ファイル sample_mnist が出来る

  make WITH_CUDA=Yes run

  とすると、MNIST データがなければダウンロードしてから
  ./sample_mnist All

  が実行される。
  sample_mnist の引数は

  LutMlp                   LUT-Networkの単純多層パーセプトロンを実行
  LutCnn                   LUT-NetworkのCNNを実行
  DenseAffine              普通の単純多層パーセプトロンを実行
  SimpleMicroMlpScratch    単純多層パーセプトロンをベタ書したサンプルを実行
  All                      上のすべてを実行

  となっている


[Windowsの場合]
  Visual C++ 2017 でビルドできる。MNISTファイルなどは手動ダウンロードが必要


【MLPの試し方】
  現在、MLP版のみ Verilogまで動作確認

  ./sample_mnist LutMlp

  を実行すると verilog ディレクトリに

  mnist_train.txt       トレーニングデータ
  mnist_test.txt        評価データ
  MnistSimpleLutMlp.v   学習済みの RTL

  が出力される

  下記を、シミュレーション実行すると、学習結果が試せる。

  tb_mnist_lut_net.v
  MnistSimpleLutMlp.v
  bb_lut.v

  なお、tb_mnist_lut_net.vtakprj が Veritak 用のプロジェクトである。

  現時点では、パスを通す確認のみで、認識率は悪いのでご了承ください。
  バイナリの場合、入出力に工夫が必要ですが、ややこしくなるので
一旦シンプルな実装にしています。

【CNNの試し方】
  現在まだVer3に移植が終わっていません。
  しばらくお待ちください m(_ _)m


