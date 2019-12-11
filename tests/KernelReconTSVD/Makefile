.PHONY: all
INC=-I../../include -I../../include/external/Eigen

all: test
test: KernelReconTSVD.cpp
	g++ -O2 $(INC) KernelReconTSVD.cpp -o KernelReconTSVD.o

# This rule tells make to delete the program
.PHONY: clean 
clean:
	rm -f KernelReconTSVD.o


