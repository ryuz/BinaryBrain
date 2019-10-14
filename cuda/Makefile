
# target
TARGET = libbbcu.a

# tools
NVCC   = nvcc
AR     = ar
RM     = rm

# flagas
CUFLAGS = -I../include --generate-code=arch=compute_35,code=sm_35 --generate-code arch=compute_50,code=sm_50 --generate-code arch=compute_61,code=sm_61 --generate-code=arch=compute_75,code=sm_75 -std=c++11 -Xcompiler -fPIC
ARFLAGS =

# sources
SRCS += Manager.cu
SRCS += LocalHeap.cu
SRCS += FrameBufferCopy.cu
SRCS += Vector.cu
SRCS += MatrixColwiseSum.cu
SRCS += MatrixColwiseMeanVar.cu
SRCS += MatrixRowwiseSetVector.cu
SRCS += MicroMlp.cu
SRCS += BinaryLut6.cu
SRCS += SparseLut.cu
SRCS += StochasticLut.cu
SRCS += StochasticMaxPooling.cu
SRCS += StochasticBatchNormalization.cu
SRCS += ShuffleModulation.cu
SRCS += BinaryToReal.cu
SRCS += Im2Col.cu
SRCS += Col2Im.cu
SRCS += MaxPooling.cu
SRCS += UpSampling.cu
SRCS += BatchNormalization.cu
SRCS += ReLU.cu
SRCS += Sigmoid.cu
SRCS += Binarize.cu
SRCS += HardTanh.cu
SRCS += Adam.cu
SRCS += LossSoftmaxCrossEntropy.cu
SRCS += AccuracyCategoricalClassification.cu

HDRS  = ../include/bbcu/bbcu.h
HDRS += ../include/bbcu/bbcu_util.h
HDRS += Common.cuh

OBJS = $(addsuffix .o, $(basename $(SRCS)))

.SUFFIXES: .cu .o

.PHONY: all
all: $(TARGET)

.PHONY: clean
clean:
	$(RM) $(TARGET) $(OBJS)

$(TARGET): $(OBJS)
	$(AR) $(ARFLAGS) rcs $(TARGET) $(OBJS)

.cu.o:
	$(NVCC) -c $(CUFLAGS) $< -o $@

$(OBJS): $(HDRS)

