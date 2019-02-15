

SRCS = MemoryTest.cpp	\
       MicroMlpAffineTest.cpp


all: test


test: $(SRCS)
	g++ -o test -mavx2 -std=c++14 -I../include -I../cereal/include $(SRCS) -lgtest_main -lgtest -lpthread
