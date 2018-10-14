if not exist train-images-idx3-ubyte.gz (
  wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
)
if not exist train-labels-idx1-ubyte.gz (
  wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
)
if not exist t10k-images-idx3-ubyte.gz (
  wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
)
if not exist t10k-labels-idx1-ubyte.gz (
  wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
)

gzip -c -d train-images-idx3-ubyte.gz  > train-images-idx3-ubyte
gzip -c -d train-labels-idx1-ubyte.gz  > train-labels-idx1-ubyte
gzip -c -d t10k-images-idx3-ubyte.gz   > t10k-images-idx3-ubyte
gzip -c -d t10k-labels-idx1-ubyte.gz   > t10k-labels-idx1-ubyte
