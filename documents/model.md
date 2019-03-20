
# BinaryBrain Version 3<br> Model説明

## 概要
本書では BinaryBrain Version 3 のモデルの概要を説明します。

## 抽象クラス
  抽象クラスは直接生成できませんが、各レイヤーの基礎となっており、操作を定義します。

### Model
  あらゆるモデルの基礎で、Forward や Backward などのAPI定義を備えます。
  また SendCommand() API を備えており、文字列によって汎用的に
各レイヤーの属性変更などを行えます。


## 活性化層
### Binarize
  バイナライズ層です。
  Forward では、0を閾値に出力を0と1に二値化します。
  Backward では hard-tanh として動作します。
  バイナリネットワークの基礎となります。

### ReLU
  ReLU です。
  Binarize から派生しており、SendCommand() にて、"binary_mode true" を送ることで
Binarize層として動作します。

### Sigmoid
  Sigmoid です。
  Binarize から派生しており、SendCommand() にて、"binary_mode true" を送ることで
Binarize層として動作します。


## 演算層
### MicroMlp
  LUT-Network の LUT に相当する部分を学習させるためのレイヤーです。
  内部は MicroMlpAffine/BatchNormalization/Activation の３層で構成されます。
  Activation は デフォルトは ReLU ですが Binary モードでは ReLUは Binalizer となります。

### MicroMlpAffine
  MicroMlp の構成要素で、入力数を6などに限定した疎結合、且つ、内部に隠れ層を備えた小さなMLP(Multi Layer Perceptron)の集合体です。
  入力数や隠れ層の数テンプレート引数で変更可能です。

### DenseAffine
  いわゆる普通の全結合のニューラルネットです。


### BatchNormalization
  バッチノーマライゼーション層です。
  バイナリ化の前にほぼ必須な要素です。


## 補助層
### Sequential
  各種の層を直列に接続して１つの層として扱えるようにします。

### LoweringConvolution
  Lowering を行い畳こみ演算を行います。
  DenseAffine を渡すと、通常のCNNになり、MicroMlp を用いたサブネットワークを渡すことで、LUT-Network での畳込みが可能です。

### RealToBinary
  実数値をバイナライズします。
  その際にframe方向に拡張して変調を掛ける(多重化)が可能です。
  現在、PWM変調と、乱数での変調を実装しており、デフォルトでPWM変調となります。
  (将来⊿Σなどの誤差蓄積機能も検討中です)

### BinaryToReal
  多重化されたバイナリ値をカウンティングして実数値を生成します。
  RealToBinary 対応しますが、こちらは時間方向だけでなく、空間方向のカウントも可能です。



