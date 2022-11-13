
# setting
export HLS_TARGET   = mnist_simple
export HLS_SOLUTION = solution_1
export DEVICE_PART  = xczu3eg-sbva484-1-i
export CLOCK_PERIOD = 4

export CSIM_OPTIONS  ?= 
export COSIM_OPTIONS ?= -trace_level all -wave_debug


# Ubuntu の場合下記が必要？
# export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu


# directories
JELLY_DIR     = ../../../../jelly
SOURCE_DIR    = src
TESTBENCH_DIR = testbench
TARGET_DIR    = $(HLS_TARGET)/$(HLS_SOLUTION)
CSIM_DIR      = $(TARGET_DIR)/csim
SYN_DIR       = $(TARGET_DIR)/syn
COSIM_DIR     = $(TARGET_DIR)/sim
IMPL_DIR      = $(TARGET_DIR)/impl
EXPORT_ZIP    = $(IMPL_DIR)/export.zip

# flags
export SOURCE_FLAGS    = -I$(SOURCE_DIR)
export TESTBENCH_FLAGS = -I$(SOURCE_DIR) -I$(TESTBENCH_DIR)

# source
SOURCES = $(SOURCE_DIR)/mnist_simple.cpp
export SOURCES

# testbanch
TESTBENCHS = $(TESTBENCH_DIR)/tb_mnist_simple.cpp
export TESTBENCHS

# rules
.PHONY: all
all: $(EXPORT_ZIP)

$(TARGET_DIR):
	-rm -rf $(HLS_TARGET)
	vitis_hls $(JELLY_DIR)/scripts/hls_create_project.tcl

$(EXPORT_ZIP): $(TARGET_DIR) $(SOURCES)
	vitis_hls $(JELLY_DIR)/scripts/hls_csynth.tcl

.PHONY: clean
clean:
	-rm -rf $(HLS_TARGET)
	-rm *.log


.PHONY: create
create: $(TARGET_DIR)

.PHONY: csynth
csynth: $(EXPORT_ZIP)

.PHONY: csim
csim: $(TARGET_DIR)
	vitis_hls $(JELLY_DIR)/scripts/hls_csim.tcl

.PHONY: cosim
cosim: $(TARGET_DIR) $(EXPORT_ZIP)
	vitis_hls $(JELLY_DIR)/scripts/hls_cosim.tcl
