CXX=nvcc
CXXFLAGS=-std=c++11 -g

LIBS=cufft curand

SOURCES = cufft_benchmark.cpp FFTSolver.cu Generator.cu Parameters.cpp gpu_utils.cu
HEADERS = FFTSolver.cuh Generator.h gpu_utils.cuh Parameters.h

.PHONY: all clean

all: cufft_benchmark # cufft_single cufft_multiple

#cufft_single: cufft_single.cu
#	$(CXX) $(CXXFLAGS) $^ -o $@ $(addprefix -l,$(LIBS))
#	
#cufft_multiple: cufft_multiple.cu
#	$(CXX) $(CXXFLAGS) $^ -o $@ $(addprefix -l,$(LIBS))

cufft_benchmark: $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $@ $(addprefix -l,$(LIBS))

clean:
	$(RM) cufft_benchmark cufft_single cufft_multiple 
