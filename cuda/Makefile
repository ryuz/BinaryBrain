

TARGET=libbbcu.a

NVCC = nvcc
CFLAGS = -I ../include

SRCS += Manager.cu
SRCS += Vector.cu
SRCS += MatrixColwiseSum.cu
SRCS += MatrixColwiseMeanVar.cu
SRCS += MatrixRowwiseSetVector.cu
SRCS += MicroMlp.cu
SRCS += BinaryToReal.cu
SRCS += Im2Col.cu
SRCS += Col2Im.cu
SRCS += MaxPooling.cu
SRCS += BatchNormalization.cu
SRCS += ReLU.cu
SRCS += Sigmoid.cu
SRCS += Binarize.cu
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