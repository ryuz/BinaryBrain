

PYTHON_PATH = /usr/include/python3.6m
CEREAL_PATH = ../cereal-1.2.2

TARGET  = core$(shell python3-config --extension-suffix)

CUARCH  = -gencode=arch=compute_35,code=sm_35 \
		  -gencode=arch=compute_50,code=sm_50 \
		  -gencode=arch=compute_60,code=sm_60 \
		  -gencode=arch=compute_61,code=sm_61 \
		  -gencode=arch=compute_75,code=sm_75

CFLAGS  = -DBB_ASSERT_EXCEPTION=1 -DBB_WITH_CEREAL=1 -DBB_WITH_CUDA=1 \
          -I$(CEREAL_PATH)/include -Iinclude -Ibinarybrain/cuda \
		  -Xcompiler -pthread -Xcompiler -mavx2 -Xcompiler -mfma \
		  -Xcompiler -fopenmp -Xcompiler -std=c++14 -Xcompiler -fPIC \
		  $(CUARCH) \
		  $(shell python3 -m pybind11 --includes)

CUFLAGS = -DBB_ASSERT_EXCEPTION=1 -DBB_WITH_CEREAL=1 -DBB_WITH_CUDA=1 \
          -Iinclude -Icuda \
		  -I$(CEREAL_PATH) \
		  $(CUARCH) \
		  -std=c++11 -Xcompiler -fPIC \
		  $(shell python3 -m pybind11 --includes)

LDFLAG = -Xcompiler -pthread -Xcompiler -fopenmp -lstdc++ -lm -lcublas


.PHONY: all
all: $(TARGET)

.PHONY: clean
clean:
	rm -f $(TARGET) core_bbcu.o core_main.o

$(TARGET): core_bbcu.o core_main.o
	nvcc -shared core_bbcu.o core_main.o $(LDFLAG) -o $(TARGET)

core_main.o: src/core_main.cpp
	nvcc $(CFLAGS) -c src/core_main.cpp -o core_main.o

core_bbcu.o: src/core_bbcu.cu
	nvcc $(CUFLAGS) -c src/core_bbcu.cu -o core_bbcu.o

