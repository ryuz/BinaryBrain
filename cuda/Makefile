

TARGET=libbbcu.a

NVCC = nvcc
CFLAGS = -I ../include

OBJS = Manager.o \
       MicroMlp.o \
	   Im2Col.o \
	   Col2Im.o \
	   MaxPooling.o \
	   BatchNormalization.o \
	   ReLU.o \
	   Binarize.o \
	   Vector.o \
       HorizontalSum.o \
       Adam.o \
       LossSoftmaxCrossEntropy.o

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
