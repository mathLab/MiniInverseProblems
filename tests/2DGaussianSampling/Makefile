.PHONY: all
INC=-I../../include -I../../include/external/Eigen

all: test
test: 2DGaussianSampling.cpp
	g++ -O2 $(INC) 2DGaussianSampling.cpp -o 2DGaussianSampling.o

# This rule tells make to delete the program
.PHONY: clean 
clean:
	rm -f 2DGaussianSampling.o


