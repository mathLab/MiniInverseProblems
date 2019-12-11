.PHONY: all
INC=-I../../include -I../../include/external/Eigen

all: test
test: DeblurTikhonov.cpp
	g++ -O2 $(INC) DeblurTikhonov.cpp -o DeblurTikhonov.o

# This rule tells make to delete the program
.PHONY: clean 
clean:
	rm -f DeblurTikhonov.o


