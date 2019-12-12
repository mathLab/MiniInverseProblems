.PHONY: all
INC=-I../../include -I../../include/external/Eigen

all: test
test: OneDblurGibbs.cpp
	g++ -O2 $(INC) OneDblurGibbs.cpp -o OneDblurGibbs.o

# This rule tells make to delete the program
.PHONY: clean 
clean:
	rm -f OneDblurGibbs.o


