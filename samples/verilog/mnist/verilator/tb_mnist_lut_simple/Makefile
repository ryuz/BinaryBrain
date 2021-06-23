
all: build run coverage

.PHONY : build
build:
	mkdir -p build && cd build && cmake ..
	cmake --build build -j

.PHONY : clean
clean:
	rm -rf build

.PHONY : mostlyclean
mostlyclean: clean
	rm -f img_*.png angle_*.png
	rm -f *.fst *.vcd
	rm -f coverage.dat
	rm -fr annotated


.PHONY : run
run:
	build/tb_verilator

.PHONY : coverage
coverage:
	verilator_coverage --annotate annotated coverage.dat

