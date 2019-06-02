
# target
TARGET  = sample-diabetes
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
SRCS  += DiabetesRegressionDenseAffine.cpp
SRCS  += DiabetesRegressionMicroMlpLut.cpp
SRCS  += DiabetesRegressionStochasticLut6.cpp

OBJS = $(addsuffix .o, $(basename $(SRCS)))

LIBS = 

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


# data
DATA_FILES  = diabetes_data.txt
DATA_FILES += diabetes_target.txt

.PHONY: dl_data
dl_data:
	python3 diabets_data.py


