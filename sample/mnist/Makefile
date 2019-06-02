
# target
TARGET  = sample-mnist
SUB_TARGETS =

# run option
RUN_OPTION = All

# default flag
DEBUG       ?= No
WITH_CUDA   ?= Yes
WITH_CEREAL ?= Yes

BBCU_PATH = ../../cuda
BBCU_LIB  = $(BBCU_PATH)/libbbcu.a

CEREAL_PATH = ../../cereal

ifeq ($(WITH_CUDA),Yes)
else
CC = g++
#CC ?= clang++
endif

CFLAGS = -mavx2 -mfma -fopenmp -std=c++14
CINCS  = -I../../include
CDEFS  = 

SRCS   = main.cpp
SRCS  += MnistStochasticLutMlp.cpp
SRCS  += MnistStochasticLutCnn.cpp
SRCS  += MnistSparseLutMlp.cpp
SRCS  += MnistSparseLutCnn.cpp
SRCS  += MnistDenseMlp.cpp
SRCS  += MnistDenseCnn.cpp
SRCS  += MnistMicroMlpLutMlp.cpp
SRCS  += MnistMicroMlpLutCnn.cpp
SRCS  += MnistCustomModel.cpp

OBJS = $(addsuffix .o, $(basename $(SRCS)))

LIBS = 

ifeq ($(DEBUG),Yes)
CFLAGS += -O0 -g
else
CFLAGS += -O2
endif

ifeq ($(WITH_CEREAL),Yes)
CDEFS      += -DBB_WITH_CEREAL
CINCS      += -I$(CEREAL_PATH)/include
endif

ifeq ($(WITH_CUDA),Yes)
CC          = nvcc
CDEFS      += -DBB_WITH_CUDA
CFLAGS     := -Xcompiler '$(CFLAGS)' -lcublas
LIBS       += $(BBCU_LIB)
SUB_TARGET += bbcu_build
endif

.SUFFIXES: .c .o

.PHONY: all
all: $(SUB_TARGET) $(TARGET)

.PHONY: clean
clean:
	rm -f $(TARGET) *.o

.PHONY: run
run: $(TARGET) train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte
	./$(TARGET) $(RUN_OPTION)

.PHONY: bbcu_build
bbcu_build:
	make -C $(BBCU_PATH)

$(TARGET): $(OBJS) $(LIBS)
	$(CC) -o $(TARGET) $(CFLAGS) $(CINCS) $(CDEFS) $(OBJS) $(LIBS)

.cpp.o:
	$(CC) $(CFLAGS) $(CINCS) $(CDEFS) -c $<

depend: $(SRCS)
	$(CC) -M $(CFLAGS) $(CINCS) $(CDEFS) $^ > $@

include depend

# mnist data
.PHONY: dl_data
dl_data: train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte

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

