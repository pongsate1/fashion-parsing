CXXFLAGS=-std=c++11 $(shell pkg-config --libs --cflags opencv)
../crf_smoothing: *.h *.cpp
	${CXX} *.cpp -o ../crf_smoothing ${CXXFLAGS} 