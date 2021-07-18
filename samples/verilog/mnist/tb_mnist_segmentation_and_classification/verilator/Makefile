

# cmake のフラグ
CMAKE_FLAGS = 


# clang があれば使う
ifeq (,$(shell which clang))
WITH_CLANG ?= No
else
WITH_CLANG ?= Yes
endif

# ninja があれば使う
ifeq (,$(shell which ninja))
WITH_NINJA ?= No
else
WITH_NINJA ?= Yes
endif

ifeq ($(WITH_CLANG),Yes)
CMAKE_FLAGS += -C ../clang-cmakeinit.cmake
endif

ifeq ($(WITH_NINJA),Yes)
CMAKE_FLAGS += -GNinja
endif



all: build run

.PHONY : build
build:
	mkdir -p build && cd build && cmake $(CMAKE_FLAGS) ..
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

