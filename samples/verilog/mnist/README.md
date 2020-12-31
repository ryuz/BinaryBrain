# MNIST Verilog シミュレーションサンプル

## ディレクトリの説明

このディレクトリが、MNISTのサンプルの Verilog 出力先ディレクトリとなっています。

シミュレーションには Simple系と CNN系の二種類があり、それぞれサンプルの実行で下記が出力されます。

Verilogシミュレーションに先立って、Python もしくは C++ で学習サンプルを実行ください。


- Simple系
  - MnistLutSimple.v  (Verilogソースコード)
  - mnist_test.txt    (シミュレーション用データ)

- CNN系
  - MnistLutCnn.v  (Verilogソースコード)
  - mnist_test_160x120.ppm  (シミュレーション用入力画像)
  - mnist_test_640x480.ppm  (シミュレーション用入力画像)

なお、学習方式が異なるサンプルでも、同じファイルに上書きしますので度のサンプルを試すかよく確認の上に利用ください。

## シミュレーション実施

シミュレーションツールには Xilinx の xsim、iverilog、veritak の3種のスクリプトを用意しています。
ですが実際には LUT-Network を iverilog で試すと極端に遅くていつまでたっても終了しませんので、基本的には xsim をご使用ください。
(なお作者は普段 Veritak-Win を使っております)


### xsim の場合

Xilinxのツールにパスが通った状態で、xsim ディレクトリで以下のいずれかを実行ください。

- Wimndows
    - run_lut_simple.bat
    - run_lut_cnn.bat

- Linux
    - run_lut_simple.sh
    - run_lut_cnn.sh


### iverilog の場合

ツールにパスが通った状態で、xsim ディレクトリで以下のいずれかを実行ください。

- Windows
    - run_lut_simple.bat
    - run_lut_cnn.bat

- Linux
    - run_lut_simple.sh
    - run_lut_cnn.sh

### Veritak-Win の場合

file_copy.bat を実行してデータファイルをカレントディレクトリにコピーした後に

- Windowsのみ
  - mnist_lut_cnn.vtakprj
  - mnist_lut_simple.vtakprj

## 結果確認

### Simple版

Simple版の場合は完了すると、認識率がコンソールに出力されます。

また vcd ファイルも出力されますので gtkwave などのツールで波形を見ることもできます。

### CNN版

シミュレーションがうまくいくと MaxPooling 層で縮小された後の画像サイズで認識結果で色付けしたものが

col_0.ppm

に出力されます。数字付近で目的の色が出ていれば正解です(黒:0, 茶:1 赤:2, 橙:3, 黄:4, 緑:5, 青:6, 紫:7, 灰:8, 白:9 )。

 pgmやppm などの [PNMファイル](https://en.wikipedia.org/wiki/Netpbm)を見るには IrfanView, gimp, MassiGra(+Susieプラグイン) などがおすすめです。



