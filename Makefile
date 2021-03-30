UNAME = $(shell uname)
ifeq ($(UNAME),Linux)
	CXX = g++-7
	#CXX1 = g++-9
	LIB_DIR = /opt/intel/compilers_and_libraries/linux/lib/intel64
	LIBS = -pthread -lm -ldl
	MKL_ROOT = /opt/intel/compilers_and_libraries/linux/mkl
	MKL_LIB_DIR = $(MKL_ROOT)/lib/intel64
endif
ifeq ($(UNAME),Darwin)
	CXX = /usr/local/bin/g++-9
	LIB_DIR = /opt/intel/compilers_and_libraries/mac/lib
	MKL_ROOT = /opt/intel/compilers_and_libraries/mac/mkl
	MKL_LIB_DIR = $(MKL_ROOT)/lib
endif

CXXFLAGS = -m64 -fopenmp -O3

LIBS = -liomp5
# LIBS = -lgomp
MKL_INC_DIR = $(MKL_ROOT)/include
MKL_LIBS = -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core
# MKL_LIBS = -lmkl_intel_lp64 -lmkl_sequential -lmkl_core 

#OBJS =	FSCU1.o trace.o
OBJS =	FSCU1.o

TARGET = FSCU1

all:	$(TARGET)

$(TARGET):	$(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) -L$(LIB_DIR) $(LIBS) -L$(MKL_LIB_DIR) $(MKL_LIBS) 

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) -I$(MKL_INC_DIR) -o $@ $<

trace.o: trace.c
	$(CXX) -O2 -c -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET)
