
TARGET = experiment-mnist

CC     = g++
#CC     = clang++
CFLAGS = -mavx2 -mfma -std=c++14 -I../../include -I../../../cereal/include -I../../../eigen

SRCS = main.cpp


$(TARGET): $(SRCS)
	$(CC) -o $(TARGET) $(CFLAGS) $(SRCS)

all: $(TARGET)

clean:
	rm -f $(TARGET) *.o


