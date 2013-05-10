# Makefile

CXX=g++-mp-4.7
CXX_FLAGS=-O3 -std=c++11

all: neural_net

neural_net: 
	${CXX} ${CXX_FLAGS} ./src/neural_net.cc -o neural_net

clean: 
	rm neural_net
