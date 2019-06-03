
all:
	nvcc -I../../include ../../cuda/MicroMlp.cu main.cpp MicroMlp_Test.cpp -o cuda_test

run:
	./cuda_test

clean:
	rm ./cuda_test
