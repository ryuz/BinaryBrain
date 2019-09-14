# target name
TARGET_NAME  = core

# tools
PYTHON  = python3
CC      = g++
#CC     = clang++
NVCC    = nvcc
AR      = g++
RM      = rm
DEPEND  = g++

# path
BB_PATH      = ../..
SRC_PATH     = ../src
PACKAGE_PATH = ../binarybrain
BBCU_PATH    = $(BB_PATH)/cuda
CEREAL_PATH  = $(BB_PATH)/cereal

# target
EXT_SUFFIX   = $(shell $(PYTHON)-config --extension-suffix)
TARGET       = $(PACKAGE_PATH)/$(TARGET_NAME)$(EXT_SUFFIX)

# control flag
DEBUG       ?= No
WITH_CUDA   ?= Yes
WITH_CEREAL ?= Yes

# flags
CFLAGS    = -pthread -mavx2 -mfma -fopenmp -std=c++14 -fPIC
CUFLAGS   = -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_75,code=sm_75
ARFLAGS   = -pthread -fopenmp -fPIC
CINCS     = -I$(BB_PATH)/cuda -I$(BB_PATH)/include $(shell $(PYTHON) -m pybind11 --includes)
CDEFS     = 

# sources
SRCS = $(SRC_PATH)/core_main.cpp

# libraries
LIBS = -lstdc++ -lm


# debug
ifeq ($(DEBUG),Yes)
CFLAGS += -g -O0 -D_DEBUG
else
CFLAGS += -g -O3
endif

# CEREAL
ifeq ($(WITH_CEREAL),Yes)
CDEFS += -DBB_WITH_CEREAL
CINCS += -I$(CEREAL_PATH)/include
endif

# CUDA
ifeq ($(WITH_CUDA),Yes)
SRCS    += $(SRC_PATH)/core_bbcu.cu
CC       = $(NVCC)
AR       = $(NVCC)
LIBS    += -lcublas
CFLAGS  := $(CUFLAGS) -Xcompiler '$(CFLAGS)'
ARFLAGS := -Xcompiler '$(ARFLAGS)'
CDEFS   += -DBB_WITH_CUDA
endif

# objects
OBJS = $(addsuffix .o, $(basename $(SRCS)))


.SUFFIXES: .c .cu .o

.PHONY: all
all: $(TARGET)

.PHONY: clean
clean:
	$(RM) -f $(TARGET) $(OBJS)

$(TARGET): $(OBJS)
	$(AR) -shared $(ARFLAGS) -o $(TARGET) $(OBJS) $(LIBS)

.cpp.o:
	$(CC) $(CFLAGS) $(CINCS) $(CDEFS) -c $< -o $@

.cu.o:
	$(NVCC) $(CUDAFLAGS) $(CINCS) $(CDEFS) -std=c++11 -Xcompiler -fPIC -c $< -o $@

depend: $(SRCS)
	$(DEPEND) -M $(CFLAGS) $(CINCS) $(CDEFS) $^ > $@

include depend

