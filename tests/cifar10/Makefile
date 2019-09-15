
# target name
TARGET      ?= test-cifar10
SUB_TARGETS ?=

# run option
RUN_OPTION ?= All

# path
BB_PATH     = ../..
BBCU_PATH   = $(BB_PATH)/cuda
CEREAL_PATH = $(BB_PATH)/cereal

BBCU_LIB    = $(BBCU_PATH)/libbbcu.a

# target control
DEBUG       ?= No
WITH_CUDA   ?= Yes
WITH_CEREAL ?= Yes

# tools
CC   = g++
# CC = clang++
NVCC = nvcc
LD   = g++

# flags
CFLAGS  = -mavx2 -mfma -fopenmp -std=c++14
CUFLAGS = -Xcompiler '$(CFLAGS)' -lcublas
LDFLAGS = $(CFLAGS)
CINCS   = -I../../include
CDEFS   = 

# sources
SRCS  = main.cpp
SRCS += Cifar10StochasticLutSimple.cpp
SRCS += Cifar10StochasticLutCnn.cpp
SRCS += Cifar10SparseLutSimple.cpp
SRCS += Cifar10SparseLutCnn.cpp
SRCS += Cifar10MicroMlpLutSimple.cpp
SRCS += Cifar10MicroMlpLutCnn.cpp
SRCS += Cifar10DenseSimple.cpp
SRCS += Cifar10DenseCnn.cpp
SRCS += Cifar10NrSparseLutCnn.cpp
SRCS += Cifar10AeSparseLutSimple.cpp
SRCS += Cifar10AeSparseLutCnn.cpp

# objs
OBJS = $(addsuffix .o, $(basename $(SRCS)))

# libs
LIBS = 


# debug
ifeq ($(DEBUG),Yes)
CFLAGS += -g -O0
CDEFS  += -D_DEBUG
else
CFLAGS += -g -O3
endif

# WITH_CEREAL
ifeq ($(WITH_CEREAL),Yes)
CDEFS  += -DBB_WITH_CEREAL
CINCS  += -I$(CEREAL_PATH)/include
endif

# WITH_CUDA
ifeq ($(WITH_CUDA),Yes)
CC          = $(NVCC)
LD          = $(NVCC)
CINCS      += -I$(CEREAL_PATH)/include
CDEFS      += -DBB_WITH_CUDA
CFLAGS     := $(CUFLAGS)
LIBS       += $(BBCU_LIB) -lcublas
SUB_TARGET += bbcu_build
endif

.SUFFIXES: .c .o

.PHONY: all
all: $(SUB_TARGET) $(TARGET)

.PHONY: clean
clean:
	rm -f $(TARGET) *.o

.PHONY: run
run: $(TARGET) $(DATA_FILES)
	./$(TARGET) $(RUN_OPTION)

.PHONY: bbcu_build
bbcu_build:
	make -C $(BBCU_PATH)

$(TARGET): $(OBJS) $(LIBS)
	$(LD) -o $(TARGET) $(LDFLAGS) $(OBJS) $(LIBS)

.cpp.o:
	$(CC) $(CFLAGS) $(CINCS) $(CDEFS) -c $<

depend: $(SRCS)
	$(CC) -M $(CFLAGS) $(CINCS) $(CDEFS) $^ > $@

include depend


# cifar-10 binary data
DATA_FILES  = cifar-10-batches-bin/test_batch.bin
DATA_FILES += cifar-10-batches-bin/data_batch_1.bin
DATA_FILES += cifar-10-batches-bin/data_batch_2.bin
DATA_FILES += cifar-10-batches-bin/data_batch_3.bin
DATA_FILES += cifar-10-batches-bin/data_batch_4.bin
DATA_FILES += cifar-10-batches-bin/data_batch_5.bin

.PHONY: dl_data
dl_data: cifar-10-binary.tar.gz
	tar zxvf cifar-10-binary.tar.gz

cifar-10-binary.tar.gz:
	wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
	tar zxvf cifar-10-binary.tar.gz


