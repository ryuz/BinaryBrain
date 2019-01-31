
TARGET = test

CC     = g++
#CC     = clang++
CFLAGS = -O3 -mavx2 -mfma -fopenmp -std=c++14 -I../../include \
         -I../../cereal/include \
         -I../../eigen

SRCS = main.cpp


$(TARGET): $(SRCS)
	$(CC) -o $(TARGET) $(CFLAGS) $(SRCS)

.PHONY: all
all: $(TARGET)

.PHONY: clean
clean:
	rm -f $(TARGET) *.o

.PHONY: run
run: $(TARGET)
	./$(TARGET)

