
# target
TARGET  = sample-cifar10
SUB_TARGETS =

# run option
RUN_OPTION = All

# default flag
DEBUG       ?= No
WITH_CUDA   ?= Yes
WITH_CEREAL ?= Yes

BBCU_PATH = ../../../cuda
BBCU_LIB  = $(BBCU_PATH)/libbbcu.a

CEREAL_PATH = ../../../cereal

ifeq ($(WITH_CUDA),Yes)
else
CC = g++
#CC ?= clang++
endif

CFLAGS = -mavx2 -mfma -fopenmp -std=c++14
CINCS  = -I../../../include
CDEFS  = 

SRCS   = main.cpp
SRCS  += Cifar10StochasticLutSimple.cpp
SRCS  += Cifar10StochasticLutCnn.cpp
SRCS  += Cifar10DifferentiableLutSimple.cpp
SRCS  += Cifar10DifferentiableLutCnn.cpp
SRCS  += Cifar10MicroMlpLutSimple.cpp
SRCS  += Cifar10MicroMlpLutCnn.cpp
SRCS  += Cifar10DenseSimple.cpp
SRCS  += Cifar10DenseCnn.cpp

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

.PHONY: mostlyclean
mostlyclean: clean
	make -C $(BBCU_PATH) clean

.PHONY: run
run: $(TARGET) $(DATA_FILES)
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

