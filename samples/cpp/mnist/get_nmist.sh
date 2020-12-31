#!/bin/bash

if [ ! -e train-images-idx3-ubyte.gz ]; then
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
fi

if [ ! -e train-labels-idx1-ubyte.gz ]; then
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
fi

if [ ! -e t10k-images-idx3-ubyte.gz ]; then
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
fi

if [ ! -e t10k-labels-idx1-ubyte.gz ]; then
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
fi

gzip -c -d train-images-idx3-ubyte.gz  > train-images-idx3-ubyte
gzip -c -d train-labels-idx1-ubyte.gz  > train-labels-idx1-ubyte
gzip -c -d t10k-images-idx3-ubyte.gz   > t10k-images-idx3-ubyte
gzip -c -d t10k-labels-idx1-ubyte.gz   > t10k-labels-idx1-ubyte
