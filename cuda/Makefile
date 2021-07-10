
# target
TARGET = libbbcu.a

# tools
NVCC   = nvcc
AR     = ar
RM     = rm

# flagas
CUFLAGS = -I../include -std=c++11 -Xcompiler -fPIC \
		   -gencode=arch=compute_52,code=sm_52 \
		   -gencode=arch=compute_53,code=sm_53 \
		   -gencode=arch=compute_60,code=sm_60 \
		   -gencode=arch=compute_61,code=sm_61 \
		   -gencode=arch=compute_62,code=sm_62 \
		   -gencode=arch=compute_70,code=sm_70 \
		   -gencode=arch=compute_72,code=sm_72 \
		   -gencode=arch=compute_75,code=sm_75
ARFLAGS =

# sources
SRCS += Manager.cu
SRCS += LocalHeap.cu
SRCS += FrameBufferCopy.cu
SRCS += ConvBitToReal.cu
SRCS += Vector.cu
SRCS += MatrixColwiseSum.cu
SRCS += MatrixColwiseMeanVar.cu
SRCS += MatrixRowwiseSetVector.cu
SRCS += MicroMlp.cu
SRCS += BinaryLut6.cu
SRCS += DifferentiableLut.cu
SRCS += StochasticLut.cu
SRCS += StochasticBatchNormalization.cu
SRCS += ShuffleModulation.cu
SRCS += RealToBinary.cu
SRCS += BinaryToReal.cu
SRCS += Im2Col.cu
SRCS += Col2Im.cu
SRCS += MaxPooling.cu
SRCS += StochasticMaxPooling.cu
SRCS += UpSampling.cu
SRCS += BatchNormalization.cu
SRCS += ReLU.cu
SRCS += Sigmoid.cu
SRCS += Binarize.cu
SRCS += HardTanh.cu
SRCS += Adam.cu
SRCS += LossSoftmaxCrossEntropy.cu
SRCS += LossMeanSquaredError.cu
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
	$(RM) -f $(TARGET) $(OBJS) depend

$(TARGET): $(OBJS)
	$(AR) $(ARFLAGS) rcs $(TARGET) $(OBJS)

.cu.o:
	$(NVCC) -c $(CUFLAGS) $< -o $@

$(OBJS): $(HDRS)

depend: $(SRCS)
	$(NVCC) -M $(CUFLAGS) $^ > $@

include depend
