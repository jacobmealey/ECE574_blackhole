NVCC=nvcc -ccbin /usr/bin/g++
NVCC_FLAGS=-allow-unsupported-compiler -lpapi -m64 -gencode arch=compute_60,code=sm_60
OBJS = main.o ray.o color.o vec3.o
TARGET = blackhole
SRCS := main.cu
INCS := $(wildcard *.h)
all: $(TARGET)

$(TARGET): blackhole.o
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) blackhole.o

blackhole.o: $(SRCS) $(INCS)
	$(NVCC) $(NVCC_FLAGS) -o blackhole.o -c main.cu 

clean:
	rm -rf $(TARGET) *.o

