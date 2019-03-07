
TARGET  = sample-mnist

BBCU_PATH = ../../cuda
BBCU_LIB  = $(BBCU_PATH)/libbbcu.a

CC     = g++
#CC     = clang++

CFLAGS = -O3 -mavx2 -mfma -fopenmp -std=c++14 \
         -DBB_WITH_CUDA \
         -I../../include \
         -I../../cereal/include \
         -I../../eigen

OBJS = main.cpp \
    	MnistSimpleMicroMlp.o \
		MnistSimpleMicroMlpScratch.o \
		MnistSimpleLutCnn.o


.SUFFIXES: .c .o

.PHONY: all
all: $(TARGET)

.PHONY: clean
clean:
	rm -f $(TARGET) *.o

.PHONY: run
run: $(TARGET) train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte
	./$(TARGET)

$(TARGET): $(OBJS) $(BBCU_LIB)
	nvcc -o $(TARGET) -Xcompiler '-O2 -fopenmp' $(OBJS) $(BBCU_LIB)

$(BBCU_LIB):
	make -C $(BBCU_PATH)

.cpp.o:
	$(CC) $(CFLAGS) -c $<

# mnist data
train-images-idx3-ubyte: train-images-idx3-ubyte.gz
	gzip -c -d train-images-idx3-ubyte.gz  > train-images-idx3-ubyte

train-labels-idx1-ubyte: train-labels-idx1-ubyte.gz
	gzip -c -d train-labels-idx1-ubyte.gz  > train-labels-idx1-ubyte

t10k-images-idx3-ubyte: t10k-images-idx3-ubyte.gz
	gzip -c -d t10k-images-idx3-ubyte.gz   > t10k-images-idx3-ubyte

t10k-labels-idx1-ubyte: t10k-labels-idx1-ubyte.gz
	gzip -c -d t10k-labels-idx1-ubyte.gz   > t10k-labels-idx1-ubyte

train-images-idx3-ubyte.gz:
	wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz

train-labels-idx1-ubyte.gz:
	wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz

t10k-images-idx3-ubyte.gz:
	wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz

t10k-labels-idx1-ubyte.gz:
	wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

