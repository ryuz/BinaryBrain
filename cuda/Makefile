

TARGET=libbbcu.a

NVCC = nvcc
CFLAGS = -I ../include

OBJS = Manager.o \
	   Im2Col.o \
       MicroMlp.o \
	   ReLU.o \
	   Binarize.o \
	   Vector.o \
       HorizontalSum.o

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
