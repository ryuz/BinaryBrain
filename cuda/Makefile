

TARGET=libbbcu.a

NVCC = nvcc
CFLAGS = -I ../include -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_75,code=sm_75 -std=c++11

SRCS += Manager.cu
SRCS += Vector.cu
SRCS += MatrixColwiseSum.cu
SRCS += MatrixColwiseMeanVar.cu
SRCS += MatrixRowwiseSetVector.cu
SRCS += MicroMlp.cu
SRCS += BinaryLut6.cu
SRCS += StochasticLut6.cu
SRCS += StochasticMaxPooling.cu
SRCS += StochasticBatchNormalization.cu
SRCS += BinaryToReal.cu
SRCS += Im2Col.cu
SRCS += Col2Im.cu
SRCS += MaxPooling.cu
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

OBJS = $(addsuffix .o, $(basename $(SRCS)))

.SUFFIXES: .cu .o

.PHONY: all
all: $(TARGET)

.PHONY: clean
clean:
	$(RM) $(TARGET) $(OBJS)

$(TARGET): $(OBJS)
	$(AR) rcs $(TARGET) $(OBJS)

.cu.o:
	$(NVCC) -c $(CFLAGS) $< -o $@

$(OBJS): $(HDRS)