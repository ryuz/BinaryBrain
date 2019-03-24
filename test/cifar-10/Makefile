
TARGET = test-cifar10


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

CFLAGS = -O2 -mavx2 -mfma -fopenmp -std=c++14
#CFLAGS = -O1 -mavx2 -mfma -std=c++14
CINCS  = -I../../include -I../../eigen
CDEFS  = 

SRCS   = main.cpp
SRCS  += Cifar10SimpleLutMlp.cpp
SRCS  += Cifar10SimpleLutCnn.cpp

OBJS = $(addsuffix .o, $(basename $(SRCS)))

LIBS = 

ifeq ($(WITH_CEREAL),Yes)
CDEFS      += -DBB_WITH_CEREAL
CINCS      += -I$(CEREAL_PATH)/include
endif

ifeq ($(WITH_CUDA),Yes)
CC          = nvcc
CDEFS      += -DBB_WITH_CUDA
CFLAGS     := -Xcompiler '$(CFLAGS)'
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

$(TARGET): $(OBJS) $(BBCU_LIB)
	$(CC) -o $(TARGET) $(CFLAGS) $(CINCS) $(CDEFS) $(OBJS) $(BBCU_LIB)

.cpp.o:
	$(CC) $(CFLAGS) $(CINCS) $(CDEFS) -c $<



# mnist data
.PHONY: dl_data
dl_data: data_batch_1.bin

data_batch_1.bin: cifar-10-binary.tar.gz
	tar zxvf cifar-10-binary.tar.gz

cifar-10-binary.tar.gz:
	wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz


