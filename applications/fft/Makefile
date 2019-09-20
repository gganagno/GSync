MACROS = ../../macros/c.linux.m4
LIBS = -lm -lpthread 
INCS =  
CC = gcc
CFLAGS = -g -fno-inline -O2 -m32 

BARFLAG = -DMBARRIER
GBARFLAG = -DGBARRIER

SPIN = -DSPINLOCK
TICKET = -DTICKETLOCK
MCS = -DMCSLOCK


objects = gspinmul.o gsync.o

all: $(objects)
	nvcc -g -G -arch=sm_60 $(objects) -std=c++11 -o out 

gspinmul.o: fft.cu
	nvcc -g -G -x cu -arch=sm_60 -std=c++11 -I. -dc $(GBARFLAG) -DGSPINLOCK -DGPU $< -o $@

gsync.o: ./../../gsync/gsync.cu 
	nvcc -g -G -x cu -arch=sm_60 -I. -dc $< -o $@


fft.cu:  fft.C4 $(MACROS)
	m4 $(MACROS) $< > $@

clean: 
	rm -f *.c *.cu *.o out


