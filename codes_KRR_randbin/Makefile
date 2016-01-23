USE_OPENBLAS=1
USE_ESSL=0
USE_OPENMP=1
OPENBLAS_LIB_PATH = -L/home/f85/lfwu/MyHomeScratch/MyLib//OpenBLAS-0.2.14/lib/ -Wl,-rpath=/home/f85/lfwu/MyHomeScratch/MyLib/OpenBLAS-0.2.14/lib/
OPENBLAS_INCL_PATH = -I/home/f85/lfwu/MyHomeScratch/MyLib/OpenBLAS-0.2.14/include/
ESSL_LIB_PATH = -L/usr/lib/ -L/opt/ibm/xlf/15.1.2/lib/ -L/opt/ibm/xlsmp/4.1.2/lib/
ESSL_INCL_PATH = -I/usr/include/

#CC        = xlc_r
CC        = g++
LINKER    = ${CC}
CFLAGS    = -std=c++11 -O3
LIBS      = -lcmatrix -lstdc++ -lm -lrt
LIB_PATHS = -L${HOME}/MyHomeScratch/Research/MachineLearning/codes_KRR_randbin/lib
INCL_PATHS = -I${HOME}/MyHomeScratch/Research/MachineLearning/codes_KRR_randbin/include

ifeq ($(USE_OPENBLAS),1)
CFLAGS += -DUSE_OPENBLAS
LIB_PATHS += ${OPENBLAS_LIB_PATH}
INCL_PATHS += ${OPENBLAS_INCL_PATH}
else ifeq ($(USE_ESSL),1)
CFLAGS += -q64 -qnoeh -DUSE_ESSL
LIB_PATHS += ${ESSL_LIB_PATH}
INCL_PATHS += ${ESSL_INCL_PATH}
endif
ifeq ($(USE_OPENMP),1)
CFLAGS += -fopenmp -DUSE_OPENMP
endif

ifeq ($(USE_OPENBLAS),1)
LIBS += -lopenblas -lpthread -lgfortran
else ifeq ($(USE_ESSL),1)
LIBS += -lesslsmp -lxlf90_r -lxl -lxlsmp -lxlfmath
else
LIBS += -llapack -lblas
endif
ifeq ($(USE_OPENMP),1)
LIBS += -lgomp
endif

all: KRR_OneVsAll_RandBin.ex 

KRR_OneVsAll_RandBin.ex: KRR_OneVsAll_RandBin.o
	${LINKER} -o KRR_OneVsAll_RandBin.ex KRR_OneVsAll_RandBin.o ${LIB_PATHS} ${LIBS}

KRR_OneVsAll_RandBin.o: KRR_OneVsAll_RandBin.cpp
	${CC} -c KRR_OneVsAll_RandBin.cpp ${CFLAGS} ${INCL_PATHS}

clean:
	rm -f *.o  *~

deepclean:
	rm -f *.o  *~ *.ex
