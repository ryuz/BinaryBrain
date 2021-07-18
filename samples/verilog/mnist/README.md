# MNIST Verilog シミュレーションサンプル

## ディレクトリの説明

このディレクトリ以下が MNISTのサンプルの Verilog ソースファイルの出力先ディレクトリとなっています。

シミュレーションに先立って、対応するサンプルプログラムを実行して学習済みの
Verilog ソースファイルを生成しておく必要があります。

Verilogシミュレーションに先立って、Python もしくは C++ で学習サンプルを実行ください。


- Simple系(tb_mnist_lut_simple ディレクトリ)
  - MnistLutSimple.v  (Verilogソースコード)
  - mnist_test.txt    (シミュレーション用データ)

- CNN系(tb_mnist_lut_cnn)
  - MnistLutCnn.v  (Verilogソースコード)
  - mnist_test_160x120.ppm  (シミュレーション用入力画像)
  - mnist_test_640x480.ppm  (シミュレーション用入力画像)

- SegmentationAndClassification (tb_mnist_segmentation_and_classification ディレクトリ)
  - MnistSegmentationAndClassification.v  (Verilogソースコード)
  - mnist_test_160x120.ppm  (シミュレーション用入力画像)
  - mnist_test_640x480.ppm  (シミュレーション用入力画像)


なお、学習方式が異なるサンプルでも、同じファイルに上書きしますので
どのサンプルを試すかよく確認の上に利用ください。


## シミュレーション実施

シミュレーションツールには verilator、xsim(Xilinx)、veritak、iverilog の4種のスクリプトを用意しています。

ただし iverilog は本システムのシミュレーションではかなり遅いようですのでお勧めしません。


### verilator の場合

verilator のツールにパスが通った状態で、 verilator ディレクトリで

```
make
```

を実行ください。

```
make clean
```

で、クリーンナップ出来ます。


### xsim の場合

Xilinxのツールにパスが通った状態で、xsim ディレクトリで以下のいずれかを実行ください。

- run_xsim.bat (Windowsの場合)
- run_xsim.sh (Linuxの場合)


### iverilog の場合

ツールにパスが通った状態で、iverilog ディレクトリで以下のいずれかを実行ください。

- run_iverilog.sh (Linuxのみ)

### Veritak-Win の場合

vertak ディレクトリにあるプロジェクトファイルを開いて実行ください。



## 結果確認

### Simple版

Simple版の場合は完了すると、認識率がコンソールに出力されます。

また vcd ファイルも出力されますので gtkwave などのツールで波形を見ることもできます。

### CNN版

シミュレーションがうまくいくと MaxPooling 層で縮小された後の画像サイズで認識結果で色付けしたものが

col_0.ppm

に出力されます。数字付近で目的の色が出ていれば正解です(黒:0, 茶:1 赤:2, 橙:3, 黄:4, 緑:5, 青:6, 紫:7, 灰:8, 白:9 )。

 pgmやppm などの [PNMファイル](https://en.wikipedia.org/wiki/Netpbm)を見るには IrfanView, gimp, MassiGra(+Susieプラグイン) などがおすすめです。


