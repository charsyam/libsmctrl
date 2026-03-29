# Set this if CUDA is installed in a different location
CUDA ?= /usr/local/cuda
# Note that CXX and CC are predefined as g++ and cc (respectively) by Make
NVCC ?= $(CUDA)/bin/nvcc
# Everything has to have -lcuda, as it's needed for libsmctrl
LDFLAGS := -lcuda -I$(CUDA)/include -L$(CUDA)/lib64

TESTS_DIR := tests
TEST_INCFLAGS := -I. -I$(TESTS_DIR)

.PHONY: clean tests all

# ----- Main Library -----
libsmctrl.so: libsmctrl.c libsmctrl.h
	$(CC) $< -shared -o $@ -fPIC $(CFLAGS) $(LDFLAGS)

# -fPIC is needed even if built as a static library, in case we are linked into
# another shared library
libsmctrl.a: libsmctrl.c libsmctrl.h
	$(CC) $< -c -o libsmctrl.o -fPIC $(CFLAGS) $(LDFLAGS)
	ar rcs $@ libsmctrl.o

# ----- Utilities -----
# Use static linking with tests to avoid LD_LIBRARY_PATH issues
libsmctrl_test_gpc_info: $(TESTS_DIR)/libsmctrl_test_gpc_info.c libsmctrl.a $(TESTS_DIR)/testbench.h
	$(CC) $< -o $@ -g $(TEST_INCFLAGS) -L. -l:libsmctrl.a $(CFLAGS) $(LDFLAGS)

# ----- Tests -----
libsmctrl_test_mask_shared.o: $(TESTS_DIR)/libsmctrl_test_mask_shared.cu $(TESTS_DIR)/testbench.h
	$(NVCC) -ccbin $(CXX) $< -c -g $(TEST_INCFLAGS)

libsmctrl_test_global_mask: $(TESTS_DIR)/libsmctrl_test_global_mask.c libsmctrl.a libsmctrl_test_mask_shared.o
	$(NVCC) -ccbin $(CXX) $< -o $@ libsmctrl_test_mask_shared.o -g $(TEST_INCFLAGS) -L. -l:libsmctrl.a $(LDFLAGS)

libsmctrl_test_stream_mask: $(TESTS_DIR)/libsmctrl_test_stream_mask.c libsmctrl.a libsmctrl_test_mask_shared.o
	$(NVCC) -ccbin $(CXX) $< -o $@ libsmctrl_test_mask_shared.o -g $(TEST_INCFLAGS) -L. -l:libsmctrl.a $(LDFLAGS)

libsmctrl_test_stream_mask_override: $(TESTS_DIR)/libsmctrl_test_stream_mask_override.c libsmctrl.a libsmctrl_test_mask_shared.o
	$(NVCC) -ccbin $(CXX) $< -o $@ libsmctrl_test_mask_shared.o -g $(TEST_INCFLAGS) -L. -l:libsmctrl.a $(LDFLAGS)

libsmctrl_test_next_mask: $(TESTS_DIR)/libsmctrl_test_next_mask.c libsmctrl.a libsmctrl_test_mask_shared.o
	$(NVCC) -ccbin $(CXX) $< -o $@ libsmctrl_test_mask_shared.o -g $(TEST_INCFLAGS) -L. -l:libsmctrl.a $(LDFLAGS)

libsmctrl_test_next_mask_override: $(TESTS_DIR)/libsmctrl_test_next_mask_override.c libsmctrl.a libsmctrl_test_mask_shared.o
	$(NVCC) -ccbin $(CXX) $< -o $@ libsmctrl_test_mask_shared.o -g $(TEST_INCFLAGS) -L. -l:libsmctrl.a $(LDFLAGS)

tests: libsmctrl_test_gpc_info libsmctrl_test_global_mask libsmctrl_test_stream_mask libsmctrl_test_stream_mask_override libsmctrl_test_next_mask libsmctrl_test_next_mask_override

all: libsmctrl.so tests

clean:
	rm -f libsmctrl.so libsmctrl.a libsmctrl.o libsmctrl_test_mask_shared.o \
	      libsmctrl_test_gpc_info libsmctrl_test_global_mask \
	      libsmctrl_test_stream_mask libsmctrl_test_stream_mask_override \
	      libsmctrl_test_next_mask libsmctrl_test_next_mask_override
