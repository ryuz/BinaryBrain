# MNIST HLS サンプル

## 事前準備

事前に学習を行ってソースコードを作成する必要があります。

samples/python/mnist/MnistDifferentiableLutHls.ipynb

を Jupyter などで実行してください。

ネットとして MnistDifferentiableLutHls.h と、テストベンチ用のデータとして mnist_test_data.h が生成されれば OK です。

また、このサンプルは本リポジトリの submodule である jelly を利用しますので、git clone 時に取得していない場合には

```
git submodule update --init --recursive
```

などのコマンドで取得ください。

また Xilinx の Vitis などのツールが必要ですので、それらがインストールされており、事前設定されているものとします。

例えば Linux なら

```
source /tools/Xilinx/Vitis/2021.2/settings64.sh 
```

などの実行で事前準備されます(OSやバージョンにより微妙に異なります)。


## 使い方

### Cシミュレーション

下記のように打つと動きます。

```
make csim
```

### 合成

下記のように打つと動きます。

```
make
```

Vivado にインポートするための zip ファイルが出来上がります。


### コシミュレーション

下記のように打つと動きます。

```
make cosim
```

デフォルトで波形確認のための GUI を起動するオプションにしております。
必要に応じて Makefile を編集ください。

