

SRCS = MemoryTest.cpp	\
       MicroMlpAffineTest.cpp


all: test


test: $(SRCS)
	g++ -o test -std=c++14 -I../include $(SRCS) -lgtest_main -lgtest -lpthread
