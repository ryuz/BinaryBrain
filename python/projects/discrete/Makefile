﻿
# Environment
PYTHON       = python3
BB_PATH      = ../../..
PACKAGE_PATH = ../../binarybrain
SRC_PATH     = $(PACKAGE_PATH)/src

EXT_SUFFIX   = $(shell $(PYTHON)-config --extension-suffix)


# target
TARGET_NAME = core
TARGET = $(PACKAGE_PATH)/$(TARGET_NAME)$(EXT_SUFFIX)
SUB_TARGETS =


# run option
RUN_OPTION = All

# default flag
DEBUG       ?= No
WITH_CUDA   ?= Yes
WITH_CEREAL ?= Yes

BBCU_PATH = $(BB_PATH)/cuda
BBCU_LIB  = $(BBCU_PATH)/libbbcu.a

CEREAL_PATH = $(BB_PATH)/cereal

ifeq ($(WITH_CUDA),Yes)
else
#CC = x86_64-linux-gnu-gcc
CC = g++
#CC ?= clang++
endif

# -pthread 
CFLAGS = -mavx2 -mfma -fopenmp -std=c++14 -fPIC
CINCS  = -I$(BB_PATH)/include $(shell $(PYTHON) -m pybind11 --includes)
CDEFS  = 

SRCS   = $(SRC_PATH)/core_main.cpp
OBJS   = $(addsuffix .o, $(basename $(SRCS)))

LIBS = -lstdc++ -lm
# -shared-libgcc

ifeq ($(DEBUG),Yes)
CFLAGS += -O0 -g
else
CFLAGS += -O3
endif

ifeq ($(WITH_CEREAL),Yes)
CDEFS      += -DBB_WITH_CEREAL
CINCS      += -I$(CEREAL_PATH)/include
endif

ifeq ($(WITH_CUDA),Yes)
CC          = nvcc
CFLAGS     := -shared -Xcompiler '$(CFLAGS)' -lcublas
CDEFS      += -DBB_WITH_CUDA
LIBS       += $(BBCU_LIB)
SUB_TARGET += bbcu_build
else
CFLAGS     := -shared $(CFLAGS)
endif

.SUFFIXES: .c .o

.PHONY: all
all: $(SUB_TARGET) $(TARGET)

.PHONY: clean
clean:
	rm -f $(TARGET) $(OBJS)

.PHONY: mostlyclean
mostlyclean: clean
	make -C $(BBCU_PATH) clean

.PHONY: bbcu_build
bbcu_build:
	make -C $(BBCU_PATH)

$(TARGET): $(OBJS)
	$(CC) -o $(TARGET) $(CFLAGS) $(OBJS) $(LIBS)

.cpp.o:
	$(CC) $(CFLAGS) $(CINCS) $(CDEFS) -c $< -o $@

depend: $(SRCS)
	$(CC) -M $(CFLAGS) $(CINCS) $(CDEFS) $^ > $@

include depend

