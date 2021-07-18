
cmake_minimum_required(VERSION 3.16)

project(tb_verilator)


find_package(verilator 4.2 REQUIRED HINTS $ENV{VERILATOR_ROOT} ${VERILATOR_ROOT})
if (NOT verilator_FOUND)
    message(FATAL_ERROR "Verilator was not found. Either install it, or set the VERILATOR_ROOT environment variable")
endif()

find_package(OpenCV REQUIRED)
if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV was not found. Either install it")
endif()


add_executable(tb_verilator ./tb_verilator.cpp)

target_include_directories(tb_verilator PUBLIC "../../../../../jelly/include")

target_include_directories(tb_verilator PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(tb_verilator PUBLIC ${OpenCV_LIBS})

set(CMAKE_CXX_FLAGS "-O3 -std=c++17")
#add_compile_definitions(WITH_OPENCV2)
target_link_libraries(tb_verilator PUBLIC "-pthread") 

verilate(tb_verilator
#   COVERAGE
#   TRACE
    INCLUDE_DIRS "."
    VERILATOR_ARGS -f verilator_cmd.txt -Os
    SOURCES ./tb_verilator.sv ../MnistLutSimple.v)
