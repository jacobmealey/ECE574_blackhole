CXX =g++
OBJS = main.o ray.o color.o vec3.o
TARGET = blackhole
CXXFLAGS = -Wall
SRCS := $(wildcard *.*pp)
OBJS := $(patsubst *.*pp, $.o, $(SRCS)) 

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $^
clean:
	rm -rf $(TARGET) *.o

