CUDA_HOME   = /Soft/cuda/12.2.2


NVCC        = nvcc
ARCH        =  -gencode arch=compute_86,code=sm_86
NVCC_FLAGS  = -O3 -Wno-deprecated-gpu-targets -I$(CUDA_HOME)/include --ptxas-options=-v -I$(CUDA_HOME)/sdk/CUDALibraries/common/inc
LD_FLAGS    = -lcudart -Xlinker -rpath,$(CUDA_HOME)/lib64 -I$(CUDA_HOME)/sdk/CUDALibraries/common/lib

EXE	        = filtrar.exe
OBJ	        = filtrar.o

default: $(EXE)

filtrar.o: filtrar.cu filter_BlackAndWhite.cu filter_Rotate.cu filter_Sobel.cu
	$(NVCC) -c -o $@ filtrar.cu  $(NVCC_FLAGS)  -I/Soft/stb/20200430  

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o *.exe 
